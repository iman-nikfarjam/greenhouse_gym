def ppm_to_gpm3(co2_ppm):
    # convert CO2 ppm to gram/m3
    return co2_ppm * 44.01 / (24.45 * 1000)


def gpm3_to_ppm(co2_gpm3):
    # convert CO2 gram/m3 to ppm
    return 24.45 * co2_gpm3 * 1000 / 44.01
