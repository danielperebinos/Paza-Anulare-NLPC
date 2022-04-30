from collections import OrderedDict

import pytest

from fiona_settings import CRS, Settings, Driver, Geometry, Type


class Collection:
    def __init__(
        self,
        driver: Driver = Driver.GeoJSON,
        schema: dict = None,
        crs: CRS = CRS.WGS84,
        encoding: str = "utf-8",
    ):
        self.driver = driver.value
        self.schema = schema
        self.crs = crs.value
        self.encoding = encoding


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def schema():
    return {'properties': {'column1': 'str'}, 'geometry': 'Point'}


def test_from_collection(schema):
    c = Collection(schema=schema)
    settings = Settings.from_collection(c)
    assert settings.schema == c.schema
    assert settings.driver == c.driver
    assert settings.crs == c.crs
    assert settings.encoding == c.encoding


def test_from_collection_with_override(schema):
    c = Collection(schema=schema)
    settings = Settings.from_collection(c, driver=Driver.GeoPackage)
    assert settings.schema == c.schema
    assert settings.driver == 'GPKG'
    assert settings.crs == c.crs
    assert settings.encoding == c.encoding


def test_properties_and_setters(settings):

    settings.crs = CRS.WGS84
    assert settings.crs == CRS.WGS84.value

    settings.driver = Driver.GeoJSON
    assert settings.driver == Driver.GeoJSON.value

    settings.encoding = 'latin1'
    assert settings.encoding == 'latin1'

    settings.geometry = Geometry.LineString
    settings.properties['column1'] = 'str'

    assert settings.schema == {
        'geometry': 'LineString',
        'properties': OrderedDict(
            column1='str'
        )
    }


def test_add_inplace(settings):
    assert len(settings.properties) == 0

    settings += ('column1', Type.str(width=25))

    assert len(settings.properties) == 1
    assert settings.schema == {
        'geometry': None,
        'properties': OrderedDict(
            column1='str:25'
        )
    }


def test_add_inplace_wrong_type(settings):
    with pytest.raises(TypeError) as exc:
        settings += ('column1',)

    assert exc.match("operand for __iadd__ must be a 2-tuple of \(column_name, column_type\)")


def test_subtract_in_place_no_column(settings):
    with pytest.raises(KeyError) as exc:
        settings -= 'column'

    assert exc.match("Property 'column' does not exist.")


def test_subtract_inplace(settings):
    settings += ('column1', Type.str())
    settings += ('column2', Type.str(width=50))

    assert len(settings.properties) == 2

    settings -= 'column1'
    assert len(settings.properties) == 1
    assert settings.properties == OrderedDict(
        column2='str:50'
    )


def test_add(settings):
    s2 = settings + ('column1', Type.str())
    assert s2 != settings
    assert len(s2.properties) == 1


def test_subtract(settings):
    settings += ('column1', Type.str())
    s2 = settings - 'column1'
    assert s2 != settings
    assert len(s2.properties) == 0
    assert len(settings.properties) == 1
