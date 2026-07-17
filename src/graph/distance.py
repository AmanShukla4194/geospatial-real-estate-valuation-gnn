"""
Distance Calculation Module.

This module provides reusable geographical distance calculation
functions for the Geospatial Real Estate Valuation project.

Currently implemented:
    • Haversine Distance

Future extensions:
    • Euclidean Distance
    • Manhattan Distance
    • KNN Distance Matrix

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from __future__ import annotations

from math import radians
from math import sin
from math import cos
from math import sqrt
from math import atan2

# =============================================================================
# Constants
# =============================================================================

EARTH_RADIUS_KM = 6371.0088

# =============================================================================
# Haversine Distance
# =============================================================================


def haversine_distance(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    """
    Compute the Haversine distance between two geographic coordinates.

    Parameters
    ----------
    latitude_1 : float
        Latitude of the first point.

    longitude_1 : float
        Longitude of the first point.

    latitude_2 : float
        Latitude of the second point.

    longitude_2 : float
        Longitude of the second point.

    Returns
    -------
    float
        Distance in kilometers.
    """

    latitude_1 = radians(latitude_1)
    longitude_1 = radians(longitude_1)

    latitude_2 = radians(latitude_2)
    longitude_2 = radians(longitude_2)

    delta_latitude = latitude_2 - latitude_1
    delta_longitude = longitude_2 - longitude_1

    a = (
        sin(delta_latitude / 2) ** 2
        + cos(latitude_1)
        * cos(latitude_2)
        * sin(delta_longitude / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a),
    )

    return EARTH_RADIUS_KM * c


# =============================================================================
# Batch Distance
# =============================================================================


def pairwise_distance(
    latitude,
    longitude,
    reference_latitude,
    reference_longitude,
):
    """
    Compute distances from multiple locations to one reference location.

    Parameters
    ----------
    latitude : iterable
    longitude : iterable
    reference_latitude : float
    reference_longitude : float

    Returns
    -------
    list[float]
    """

    distances = []

    for lat, lon in zip(latitude, longitude):

        distances.append(
            haversine_distance(
                lat,
                lon,
                reference_latitude,
                reference_longitude,
            )
        )

    return distances


# =============================================================================
# Validation
# =============================================================================


def validate_coordinates(
    latitude: float,
    longitude: float,
) -> bool:
    """
    Validate latitude and longitude.

    Returns
    -------
    bool
    """

    return (
        -90 <= latitude <= 90
        and
        -180 <= longitude <= 180
    )


# =============================================================================
# Demonstration
# =============================================================================

if __name__ == "__main__":

    point_a = (47.5112, -122.2570)

    point_b = (47.7210, -122.3190)

    distance = haversine_distance(
        point_a[0],
        point_a[1],
        point_b[0],
        point_b[1],
    )

    print("=" * 60)
    print("Haversine Distance Test")
    print("=" * 60)
    print(f"Distance : {distance:.3f} km")
    print("=" * 60)