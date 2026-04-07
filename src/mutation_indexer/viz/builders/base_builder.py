import abc
import copy
import logging
from typing import ClassVar, Self

from pyspark import sql
from pyspark.sql.types import ArrayType, BooleanType, MapType, StructType

from mutation_indexer import es_utils
from mutation_indexer.builders import utils
from mutation_indexer.constants import app, build

logging.basicConfig(format=app.LOG_FORMAT)


def get_all_boolean_paths(mapping):
    """Find all the boolean field in mapping and return the paths

    Args:
        mapping: dict of mapping types

    Returns:
        list of path, each path is a list of field names
    """
    res = []

    def helper(node, path=None):
        if path is None:
            path = []

        for key, value in node["properties"].items():
            if value.get("type") == "boolean":
                res.append([*path, key])
            elif "properties" in value:
                helper(value, [*path, key])

    helper(mapping)
    return res


def cast_booleans(df, mapping):
    """Ensure all the boolean fields in data frame are booleans before save to ES

    Args:
        df: pyspark dataframe to cast boolean
        mapping: Dict of mapping types

    Returns:
        pyspark dataframe with boolean field casted
    """
    paths = get_all_boolean_paths(mapping)
    schema = copy.deepcopy(df.schema)
    for path in paths:
        field = None
        for node in path:
            if field is None:
                field = schema[node]
            elif isinstance(field.dataType, StructType):
                field = field.dataType[node]
            elif isinstance(field.dataType, ArrayType):
                field = field.dataType.elementType[node]
            elif isinstance(field.dataType, MapType):
                raise ValueError("Unsupported Property Type")
        field.dataType = BooleanType()

    select_expr = [df[f.name].cast(f.dataType) for f in schema.fields]
    return df.select(*select_expr)


class BaseBuilder(abc.ABC):
    """
    BaseBuilder contains the structure necessary for a Builder object.
    """

    index_name: ClassVar[str]
    id_field: ClassVar[str]

    def __init__(self, config, sql_context):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sql_context = sql_context
        self.debug = config.debug
        self.mappings_loader = es_utils.MappingsLoader()

    @abc.abstractmethod
    def build(self, **kwargs: sql.DataFrame) -> Self:
        """
        Contains the ETL logic to construct a spark dataframe of
        the same structure as the required output index.
        """
        pass

    def load(self):
        """
        Responsible for loading the dataframe resulting from :func:`build`
        into a destination, usually Elasticsearch.
        """
        index = self.config.indices[self.index_name]
        mapper = self.mappings_loader.load_mapper(build.IndexType[self.index_name.upper()])

        self.log(f"Creating {index} index")
        response = self.config.es.indices.create(
            index=index, mappings=mapper.mappings, settings=mapper.settings,
            timeout="60s",
            master_timeout="60s"
        )
        self.log(response)

        self.log(f"Repartitioning {self.index_name}")
        df = getattr(self, self.index_name).repartition(
            self.config.df_repartition, self.id_field
        )

        df = cast_booleans(df, mapper.mappings)
        self.log(f"Exporting {self.index_name} index to {index}")
        df.coalesce(self.config.df_coalesce).write.format(
            "org.elasticsearch.spark.sql"
        ).option("es.nodes", self.config.es_nodes).option(
            "es.net.http.auth.user", self.config.source_es_user
        ).option("es.net.http.auth.pass", self.config.es_pass).option(
            "es.net.ssl", self.config.es_use_ssl
        ).option(
            "es.net.ssl.cert.allow.self.signed", self.config.disable_es_verify_certs
        ).option("es.nodes.wan.only", "false").option(
            "es.nodes.resolve.hostname", "false"
        ).option("es.resource.write", index).option("es.http.timeout", "1h").option(
            "es.http.retries", "-1"
        ).option("es.batch.write.retry.count", "-1").option(
            "es.batch.write.retry.wait", "10m"
        ).option("es.batch.size.bytes", self.config.batch_size_bytes).option(
            "es.batch.size.entries", self.config.batch_size_entries
        ).option("es.batch.write.refresh", False).option("es.mapping.id", self.id_field).save(
            index
        )
        self.log(f"Finished exporting {self.index_name} index to {index}")

        df.unpersist()

    def truncate_df_at_percentile(
        self,
        df_to_truncate,
        field,
        percentile_threshold,
    ):
        """
        Truncates df_to_truncate to remove rows
        where field > percentile_threshold
        """
        return utils.filter_arrays_by_relative_size(
            df_to_truncate, field, percentile_threshold
        )

    def load_raw(self, path=None):
        """
        Loads the computed index's dataframe, if it exists, and return it,
        returns None it does not
        """
        if path is None:
            path = self.config.get_raw_output_path(self.index_name)
        try:
            self.logger.info(f"Using existing index from {path}")
            df = self.sql_context.read.load(path)
            return df
        except Exception:
            self.logger.info(f"Couldn't find file at {path}")
            return None

    def write(self, path=None):
        """
        Writes the built dataframe to a json file at path
        """
        if not self.config.output_raw == "write":
            self.logger.info("Will not write raw output to s3")
            return

        if path is None:
            path = self.config.get_raw_output_path(self.index_name)

        df = getattr(self, self.index_name, None)
        assert df is not None, "Builder does not have index_name attribute"

        # Repartition by the id into number of partitions specified in config
        id_field = getattr(self, self.id_field, None)
        if id_field:
            df = df.repartition(self.config.df_repartition, id_field).write
        else:
            df = df.repartition(self.config.df_repartition).write
            df = df.mode("overwrite")
        self.logger.info(f"Saving {self.index_name} to {path}")
        df.json(path)

    def log(self, string):
        """
        Handles Builder logging.
        """
        self.logger.info(string)

    def log_count(self, dataframe):
        """
        Logs dataframe count if in Debug mode
        """
        if self.debug:
            self.log(f"Count: {dataframe.count()}")
