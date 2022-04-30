"""
Copyright 2020 Tom Caruso

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import typing
import copy

from collections import OrderedDict, UserDict
from enum import Enum


class CRS(Enum):
    WGS84 = {"init": "epsg:4326", "no_defs": True}
    NAD83 = {"init": "epsg:4269", "no_defs": True}
    WebMercator = {"init": "epsg:3857", "no_defs": True}


class Driver(Enum):
    GeoJSON = "GeoJSON"
    GeoPackage = "GPKG"
    MapInfoFile = "MapInfo File"
    Shapefile = "Esri Shapefile"


class Geometry(Enum):
    Point = "Point"
    LineString = "LineString"
    Polygon = "Polygon"
    MultiPoint = "MultiPoint"
    MultiLineString = "MultiLineString"
    MultiPolygon = "MultiPolygon"


def _allow_width(s):
    def size(width=None):
        if width:
            return f"{s}:{width}"
        return s

    return size


class Type:
    str = _allow_width("str")
    float = _allow_width("float")
    int = _allow_width("int")
    date = "date"
    datetime = "datetime"
    time = "time"


def _value_or_self(v):
    return getattr(v, "value", v)


class Settings(UserDict):
    def __init__(
        self,
        driver: Driver = None,
        schema: typing.Dict[str, str] = None,
        crs: CRS = None,
        encoding: str = "utf-8",
    ):
        super().__init__()
        seed = copy.deepcopy(schema) or {}
        self.update(
            {
                "driver": _value_or_self(driver),
                "schema": {
                    "properties": seed.get("properties", OrderedDict()),
                    "geometry": _value_or_self(seed.get("geometry", None)),
                },
                "crs": _value_or_self(crs),
                "encoding": encoding,
            }
        )

    @classmethod
    def from_collection(cls, collection, **kwargs):
        k = {
            "driver": kwargs.get('driver') or collection.driver,
            "schema": kwargs.get('schema') or collection.schema,
            "crs": kwargs.get('crs') or collection.crs,
            "encoding": kwargs.get('encoding') or collection.encoding
        }
        return cls(**k)

    @property
    def schema(self):
        return self["schema"]

    @property
    def properties(self):
        return self.schema["properties"]

    @property
    def geometry(self) -> str:
        return self.schema["geometry"]

    @geometry.setter
    def geometry(self, value: Geometry):
        self.schema["geometry"] = _value_or_self(value)

    @property
    def crs(self) -> CRS:
        return self["crs"]

    @crs.setter
    def crs(self, value: CRS):
        self["crs"] = _value_or_self(value)

    @property
    def driver(self) -> Driver:
        return self["driver"]

    @driver.setter
    def driver(self, value: Driver):
        self["driver"] = _value_or_self(value)

    @property
    def encoding(self):
        return self["encoding"]

    @encoding.setter
    def encoding(self, value):
        self["encoding"] = _value_or_self(value)

    def __iadd__(self, other):
        if isinstance(other, tuple) and len(other) == 2:
            self.properties[other[0]] = other[1]
            return self
        else:
            raise TypeError("operand for __iadd__ must be a 2-tuple of (column_name, column_type)")

    def __add__(self, other):
        s = Settings(**self)
        s += other
        return s

    def __isub__(self, other):
        if other not in self.properties:
            raise KeyError(f"Property '{other}' does not exist.")
        del self.properties[other]
        return self

    def __sub__(self, other):
        s = Settings(**self)
        s -= other
        return s
