import math


def humidity_abs_to_rh(temperature_celsius: float, absolute_humidity: float) -> float:
    """Turns Absolute Humidity in [g/m3] to Relative Humidity"""
    relative_humidity = absolute_humidity / (
        (6.112 * math.exp(17.67 * temperature_celsius / (temperature_celsius + 243.5)) * 18.02)
        / ((273.15 + temperature_celsius) * 0.08314)
    )
    return max(0.0, min(relative_humidity, 1.0))


def humidity_rh_to_abs(temperature_celsius: float, relative_humidity: float) -> float:
    """Turns Relative Humidity to Absolute Humidity in [g/m3]"""
    return (
        relative_humidity
        * (6.112 * math.exp(17.67 * temperature_celsius / (temperature_celsius + 243.5)) * 18.02)
        / ((273.15 + temperature_celsius) * 0.08314)
    )


def saturated_vapor_density(temperature: float) -> float:
    """
    Calculate saturated vapor density g/m3
    For types see http://hyperphysics.phy-astr.gsu.edu/hbase/Kinetic/watvap.html
    Saturated vapor density table has been fitted to 3rd degree polynome using Excel
    Temperate: degrees
    vapor density: gr/m3
    """
    return 0.0006 * temperature**3 - 0.0021 * temperature**2 + 0.3322 * temperature + 5.8649


if __name__ == "__main__":
    for temp in range(101):
        print(saturated_vapor_density(temp))

    temp = 25
    for rh_percentage in range(101):
        rh = rh_percentage / 100
        abs_h = humidity_rh_to_abs(temp, rh)
        # print(temp, rh_percentage, abs_h)
        rh_after = humidity_abs_to_rh(temp, abs_h)
        print(rh, abs_h, rh_after)
        assert abs(rh - rh_after) < 0.01
