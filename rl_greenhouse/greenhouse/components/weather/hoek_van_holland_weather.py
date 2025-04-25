import datetime
import pickle
import os

from rl_greenhouse.greenhouse.components.base import WeatherComponent
from rl_greenhouse.greenhouse.types import Weather, Info


class HoekVanHollandWeatherStation(WeatherComponent):
    """
    Weather station that returns a given weather pattern

    KNMI Data for Hoek van Holland from the 2001 to 2020 period.
    """

    weather = None

    def __init__(self, hyper_parameters):
        super().__init__(hyper_parameters, parameters_used={"start_date": "Start date of the simulation."})
        self.start_date = hyper_parameters["start_date"]
        self.start_date_randomization = hyper_parameters["start_date_randomization"]
        self.start_date_offset = hyper_parameters["start_date_offset"]

        path = os.path.join(os.path.dirname(__file__), "weather_hoek_van_holland.pickle")
        if not HoekVanHollandWeatherStation.weather:
            with open(path, "rb") as handle:
                HoekVanHollandWeatherStation.weather = pickle.load(handle)

    def step(self, info: Info) -> Weather:
        """
        Calculates the resulting weather from applying the effect of this subsystem.

        :param info: Information about the global state of the system.
        """
        # Weather for time + 1 otherwise par light is calculated with old weather information which makes the
        # par light value incorrect. This shift fixes this by artificially first taking the weather w(t)
        # and then going into the update loop to predict s(t) as s(t) is dependent on w(t).
        return self.weather[info.date_time + datetime.timedelta(hours=1)]

    def reset(self, start_date: datetime.datetime):
        """
        Reset this subsystem.
        """
        # Weather for time + 1 otherwise par light is calculated with old weather information which makes the
        # par light value incorrect. This shift fixes this by artificially first taking the weather w(t)
        # and then going into the update loop to predict s(t) as s(t) is dependent on w(t).
        start_date = start_date + datetime.timedelta(hours=1)
        return self.weather[start_date]


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv("weather_hoek_van_holland_2001_2022.csv")
    weather = {}
    for idx, row in data.iterrows():
        measure_date = pd.to_datetime(row["date"])
        measure_hour = round(float(row["hour"])) % 24
        measure_datetime = datetime.datetime(
            year=measure_date.year, month=measure_date.month, day=measure_date.day, hour=measure_hour
        )
        if measure_datetime in weather.keys():
            raise KeyError(f"Duplicates found for key {measure_datetime}")

        weather[measure_datetime] = Weather(
            illumination_out=float(row["illumination_out"]),
            relative_humidity_out=float(row["humidity_out"]),
            temperature_out=float(row["temperature_out"]),
            wind_out=float(row["wind_out"]),
        )

    with open("weather_hoek_van_holland.pickle", "wb") as handle:
        pickle.dump(weather, handle, protocol=pickle.HIGHEST_PROTOCOL)
