"""
icyrivermodel.py

Josie Arcuri

July 2022

"""

import numpy as np
from landlab import RasterModelGrid, imshow_grid
import matplotlib.pyplot as plt
import os
import imageio
from scipy import interpolate


class Profile:
    """class for bank profile objects"""

    def __init__(self, x, z, T, t):
        """
        initialize profile object

        Parameters
        ----------
        x: array
            x-coordinates of nodes, m
        z: array
            z-coordinates of nodes, m
        T: array
            riverbank surface temperature at x,z coordinate, degrees C
        t: float
            representative time - or age - of profile, days
        """
        self.x = x
        self.z = z
        self.T = T
        self.age = t


class IcyRiver:
    def __init__(self, params={}):
        # unpack parameters

        # timing
        self.final_time = params["run_duration"]
        self.time = 0
        self.dx = params["dx"]
        self.dt = params["dt"]
        self.save_dt = params["save_dt"]

        # air temperature
        self.air_temperature__mean = params["air_temperature__mean"]
        self.air_temperature__amplitude = params["air_temperature__amplitude"]
        self.air_temperature__period = params["air_temperature__period"]
        self.air_temp = self.air_temperature__mean - self.air_temperature__amplitude / 2

        # riverbank geometry
        self.river_stage = params["initial_river_stage"]
        self.river_half_width = params["river_half_width"]
        self.bank_height = params["bank_height"]
        self.bank_width = params["bank_width"]
        self.riverbed_z = 0
        self.riverbed_temp = params["riverbed_temp"]
        self.morph_factor = params["morph_factor"]
        self.dzdxcrit = 4
        # erosion rate constants
        self.alpha_are = params["subaerial_erosion"]

        # river temperature
        self.river_temp = 273.15
        self.river_temp_max = params["water_temperature__max"]

        self.trib = params["trib_doy"]  # 140 doy
        self.tfreeze = params["tfreeze_doy"]  # 310
        self.thighstage = self.trib + 31

        # material properties
        self.L_ice = 334000  # J/kg
        self.C_ice = 2000  # J/kgK
        self.C_water = 4000  # J/kgK
        self.rho_ice = 917  # kg/m3
        self.k_ice = 2 * (60 * 60 * 24)  # J/daymK
        self.k_perma = 1.5 * (60 * 60 * 24)  # J/daymK
        self.rho_perma = 2200  # kg/m3
        self.melt_temp = 273.15  # K
        self.W = params["soil_percent_water"] / 100  # .3
        # create landlab grid
        n_rows = (self.bank_height + abs(params["lower_bound__depth"]) + 1) / self.dx
        n_cols = (self.bank_width + self.river_half_width) / self.dx
        self.grid = RasterModelGrid(
            (n_rows, n_cols),
            xy_spacing=self.dx,
            xy_of_lower_left=(0, params["lower_bound__depth"]),
            xy_of_reference=(0, params["lower_bound__depth"]),
        )
        # add fields for heat diffusion
        self.T = self.grid.add_zeros("temperature", at="node")
        self.qT = self.grid.add_zeros("heat_flux", at="link")

        self.erosion = self.grid.add_zeros("eroded", at="node")
        # divide core nodes into air, river, riverbed

        self.river_nodes = np.where(
            np.logical_and(
                self.grid.x_of_node >= self.bank_width,
                np.logical_and(
                    self.grid.y_of_node <= self.river_stage,
                    self.grid.y_of_node > self.riverbed_z,
                ),
            )
        )[0][:]

        self.air_nodes = np.where(
            np.logical_or(
                self.grid.y_of_node == np.max(self.grid.y_of_node),
                np.logical_and(
                    self.grid.x_of_node >= self.bank_width,
                    self.grid.y_of_node > self.river_stage,
                ),
            )
        )[0][:]

        self.riverbed_nodes = np.where(
            np.logical_and(
                self.grid.y_of_node == self.riverbed_z,
                self.grid.x_of_node >= self.bank_width,
            )
        )[0][:]
        # set non permafrost core nodes to fixed value boundaries, and at grid edges
        self.grid.status_at_node[self.river_nodes] = self.grid.BC_NODE_IS_FIXED_VALUE
        self.grid.status_at_node[self.riverbed_nodes] = self.grid.BC_NODE_IS_FIXED_VALUE
        self.grid.status_at_node[self.air_nodes] = self.grid.BC_NODE_IS_FIXED_VALUE

        self.grid.set_fixed_value_boundaries_at_grid_edges(
            True, True, True, True, value_of="temperature"
        )

        # set temperature at boundaries
        self.T[:] = params["lower_bound__temperature"]
        self.T[self.riverbed_nodes] = self.riverbed_temp
        self.T[self.river_nodes] = self.river_temp
        self.T[self.air_nodes] = self.air_temp

        # set initial profile
        self.profile_dx = params["profile_dx"]  # m
        self.profile_npts = np.ceil(
            (self.bank_width + self.bank_height + self.river_half_width)
            / self.profile_dx
        )
        x, z = self.create_initial_profile()
        # grab temperature at bnklne nodes
        T, node_ids = self.map_temp_to_bank(x, z)
        # save first profile object
        self.profiles = []
        self.profiles.append(Profile(x, z, T, 0))

    def run(self):
        """Run simulation from start to finish"""

        while self.time <= self.final_time:
            # keep time
            self.time += self.dt
            # update bcs
            self.set_air_temperature()
            self.set_river_temperature()

            self.T[self.air_nodes] = self.air_temp
            self.T[self.river_nodes] = self.river_temp
            self.T[self.riverbed_nodes] = self.riverbed_temp
            # calculate temperature on landlab grid
            grad = self.grid.calc_grad_at_link(self.T)
            self.qT[self.grid.active_links] = (
                -((self.k_perma) / (self.rho_perma * self.C_ice))
                * grad[self.grid.active_links]
            )
            dTdt = -self.grid.calc_flux_div_at_node(self.qT)
            self.T[self.grid.core_nodes] += dTdt[self.grid.core_nodes] * self.dt

            # map temperature to profile
            x = self.profiles[-1].x
            z = self.profiles[-1].z

            if np.round(self.time, 2) % self.save_dt == 0:
                self.plot_riverbank_temp()

            (T, node_ids) = self.map_temp_to_bank(x, z)

            # calculate erosion
            if self.time > 365:
                E = self.calc_thermal_erosion(x, z, T, dTdt[node_ids])
                E += self.calc_slumping(x, z)
            else:
                E = np.zeros_like(x)

            # update profile
            x_eroded = x + E
            self.profiles.append(profile(x_eroded, z, T, self.time))
            # redefine grid boundaries
            self.map_erosion_to_grid(x_eroded, z, E)

    def map_erosion_to_grid(self, x, z, E):
        """
        re-allocate boundary conditions, update grid
        Parameters
        ----------
        x: array
            x-coordinates of profile nodes, m
        z: array
            z-coordinates of profile nodes, m
        E: array
            riverbank surface temperature at x,z coordinate, degrees C

        """
        bottom_bank_x = x[np.where(z != self.riverbed_z)[0][-1]]
        dz = np.concatenate((np.diff(z), [0]))
        dx = np.concatenate((np.diff(x), [0]))
        bluff_id = np.where(x > np.min(x[z < self.bank_height]))[0]
        bed_grid_nodes = np.where(
            (self.grid.y_of_node == self.riverbed_z)
            & (self.grid.node_is_boundary(np.arange(len(self.grid.y_of_node))) == False)
            & (self.grid.x_of_node > bottom_bank_x)
        )[0]
        for i in bluff_id:
            node_i = np.where(
                (self.grid.x_of_node >= (x[i] - (self.dx / 2)))
                & (self.grid.y_of_node >= (z[i] - (self.dx / 2)))
            )[0][0]

            x_node = self.grid.x_of_node[node_i]
            z_node = self.grid.y_of_node[node_i]
            row = np.where(
                (self.grid.y_of_node == z_node) & (self.grid.x_of_node >= x_node)
            )[0][:]
            dist_to_bank = x_node - x[i]
            dist_to_riv = z_node - self.river_stage

            # get first boundary node to the right
            #
            new_nodes = row[
                np.where(
                    (self.grid.x_of_node[row] - x[i] >= (self.dx / 2))
                    & (self.grid.node_is_boundary(row) == False)
                )
            ]
            if len(new_nodes) > 0:
                if dist_to_riv > 0:

                    self.grid.status_at_node[
                        new_nodes
                    ] = self.grid.BC_NODE_IS_FIXED_VALUE
                    self.air_nodes = np.concatenate((self.air_nodes, new_nodes))

                #    if (dist_to_riv <= 0) & (z_node <= self.riverbed_z + self.dx):

                if (dist_to_riv <= 0) & (z_node > self.riverbed_z):
                    self.grid.status_at_node[
                        new_nodes
                    ] = self.grid.BC_NODE_IS_FIXED_VALUE
                    self.river_nodes = np.concatenate((self.river_nodes, new_nodes))
        if np.sum(E[np.where((E <= -self.dx) & (z > self.river_stage))[0]]) < 0:

            hinge = np.where((E <= -self.dx) & (z > self.river_stage))[0][-1] - 1

            x_hinge = x[hinge]
            z_hinge = z[hinge]

            chunk_air = np.where(
                (self.grid.x_of_node > (x_hinge + self.dx / 2))
                & (self.grid.y_of_node > (self.river_stage + self.dx / 2))
                & (
                    self.grid.node_is_boundary(np.arange(len(self.grid.y_of_node)))
                    == False
                )
            )[0][:]

            self.grid.status_at_node[chunk_air] = self.grid.BC_NODE_IS_FIXED_VALUE
            self.air_nodes = np.concatenate((self.air_nodes, chunk_air))

        if len(bed_grid_nodes) > 0:
            self.grid.status_at_node[bed_grid_nodes] = self.grid.BC_NODE_IS_FIXED_VALUE
            self.riverbed_nodes = np.concatenate((self.riverbed_nodes, bed_grid_nodes))

        self.T[self.air_nodes] = self.air_temp
        self.T[self.river_nodes] = self.river_temp
        self.T[self.riverbed_nodes] = self.riverbed_temp

    def map_temp_to_bank(self, x, z):
        """
        find temperature values on landlab grid at x,z profile coordinates
        Parameters
        ----------
        x: array
            x-coordinates of profile nodes, m
        z: array
            z-coordinates of profile nodes, m
        E: array
            riverbank surface temperature at x,z coordinate, degrees C

        Returns
        --------
        T: array
            temperatures at profile nodes
        node_ids:
            grid node ids corresponding to profile nodes
        """
        # find core node nearest x,z, coordinate
        dz = np.concatenate(([0], np.diff(z)))
        dx = np.concatenate(([0], np.diff(x)))
        bluff_id = np.where((z != self.bank_height) & (z != self.riverbed_z))[0]
        flat_id = np.where((z == self.bank_height) & (z == self.riverbed_z))[0]

        # for x in columns,
        node_ids = np.ones(len(x), dtype="int")
        for i in bluff_id:
            node_ids[i] = int(self.grid.find_nearest_node((x[i], z[i])))

            if self.grid.node_is_boundary(int(node_ids[i])) == True:
                # look left
                node_ids[i] -= 1
        for i in flat_id:
            node_ids[i] = int(self.grid.find_nearest_node((x[i], z[i])))
            # look down
            if self.grid.node_is_boundary(int(node_ids[i])) == True:
                node_ids[i] = int(
                    self.grid.find_nearest_node((x[i], z[i] - self.grid.dx))
                )

        # find temperature of that node
        T = np.asarray(self.T[node_ids])
        return (T, node_ids)

    def calc_thermal_erosion(self, x, z, T, T_div):
        """
        commit subaerial and subaqueous melting

        find temperature values on landlab grid at x,z profile coordinates
        Parameters
        ----------
        x: array
            x-coordinates of profile nodes, m
        z: array
            z-coordinates of profile nodes, m
        E: array
            riverbank surface temperature at x,z coordinate, degrees C

        Returns
        --------
        erosion: array
            magnitudes of erosion at profile nodes


        """

        dzdx = np.concatenate((np.diff(z) / np.diff(x), [0]))
        below_water = np.where((z > self.riverbed_z) & (z <= self.river_stage))[0][:]
        frozen_array = np.zeros_like(below_water)
        frozen_array[T[below_water] < self.melt_temp] = 1
        L_array = np.ones_like(below_water) * self.L_ice
        L_array[T[below_water] >= self.melt_temp] = 0
        C_array = np.zeros_like(below_water) + self.C_water
        C_array[T[below_water] <= self.melt_temp] = self.C_ice
        erosion = np.zeros_like(z)
        dT = np.abs(np.asarray(self.profiles[-1].T)[below_water] - T[below_water])

        erosion[below_water] = (
            -(
                frozen_array
                * self.morph_factor
                * self.k_perma
                * ((self.river_temp - self.melt_temp) + self.dx * T_div[below_water])
                / (
                    L_array * self.W * self.rho_ice
                    + (
                        self.rho_perma
                        * C_array
                        * np.abs((self.melt_temp - T[below_water]))
                    )
                )
            )
            * self.dt
        )
        # if np.sum(erosion) < 0:
        #    print("melted! at DOY:" + str(int(self.time) % 365))

        # subaerial melting
        above_water_bot = np.where(z > self.river_stage)[0][-1]
        above_water_top = np.where(z == self.bank_height)[0][-1]

        erosion[above_water_bot:above_water_top] = (
            -self.alpha_are * self.dt * (self.air_temp - self.melt_temp)
        )
        erosion[erosion > 0] = 0

        return erosion

    def calc_slumping(self, x, z):
        """
        shear off overhang when bank extends past 45 degrees

        Parameters
        ----------
        x: array
            x-coordinates of profile nodes, m
        z: array
            z-coordinates of profile nodes, m

        Returns
        --------
        erosion: array
            magnitudes of erosion at profile nodes


        """
        erosion = np.zeros(len(x))
        hinge = np.where(
            x
            == np.min(x[np.where((z > self.riverbed_z) & (z < self.bank_height))[0][:]])
        )[0][0]

        top = np.where(z == self.bank_height)[0][-1]
        overhang = (z[top] - z[hinge]) / (x[top] - x[hinge])
        # print((x[hinge] - x[top]))
        if (overhang <= self.dzdxcrit) & (overhang > 0):

            notch = np.where((x >= x[hinge]) & (z > z[hinge]))[0][:]
            # print(len(notch))

            buddy = x
            erosion[notch] = x[hinge] - x[notch]
            erosion[erosion > 0] = 0

            # erosion[erosion > 0] = 0
            print("slump! at DOY: " + str(int(self.time % 365)))
        return erosion

    def create_grid(self):
        """
        create riverbank landlab grid
        """
        n_rows = (self.bank_width + self.river_half_width) / self.dx
        n_cols = (self.bank_height + self.lower_bound__depth + 1) / self.dx
        self.grid = RasterModelGrid((n_rows, n_cols), xy_spacing=self.spacing)

    def create_initial_profile(self):
        """
        create profile object - rectangle

        """
        # top
        x_top = np.linspace(
            0, (self.bank_width), int(self.bank_width / self.profile_dx)
        )
        z_top = np.ones_like(x_top) * (self.riverbed_z + self.bank_height)
        # bluff
        z_bluff = np.linspace(
            self.bank_height + self.riverbed_z,
            self.riverbed_z,
            int(self.bank_height / self.profile_dx),
        )

        x_bluff = np.ones_like(z_bluff) * self.bank_width
        # bed
        x_bed = np.arange(
            self.bank_width + self.dx,
            (self.bank_width + self.river_half_width) - self.dx,
            self.profile_dx,
        )
        z_bed = np.ones_like(x_bed) * self.riverbed_z

        x = np.concatenate((x_top, x_bluff, x_bed))
        z = np.concatenate((z_top, z_bluff, z_bed))

        return x, z

    def set_river_temperature(self):
        """
        """
        trib_airtemp = np.mean(
            (
                get_air_temperature(
                    self.trib,
                    mean=self.air_temperature__mean,
                    amplitude=self.air_temperature__amplitude,
                    period=self.air_temperature__period,
                ),
                self.melt_temp,
            )
        )
        tfreeze_airtemp = np.mean(
            (
                get_air_temperature(
                    self.tfreeze,
                    mean=self.air_temperature__mean,
                    amplitude=self.air_temperature__amplitude,
                    period=self.air_temperature__period,
                ),
                self.melt_temp,
            )
        )
        if (self.time % 365) >= self.trib and (self.time % 365) <= self.thighstage:
            self.river_temp = trib_airtemp + (self.river_temp_max - trib_airtemp) * (
                (int(self.time % 365) - self.trib) / (self.thighstage - self.trib)
            )

            self.riverbed_temp = self.melt_temp
        if (self.time % 365) > self.thighstage and (self.time % 365) <= self.tfreeze:
            self.river_temp = tfreeze_airtemp + (
                self.river_temp_max - tfreeze_airtemp
            ) * (
                (self.tfreeze - int(self.time % 365))
                / ((self.tfreeze - self.thighstage))
            )

            self.riverbed_temp = self.melt_temp
        else:
            self.river_temp = np.mean((self.air_temp, self.melt_temp))
            self.riverbed_temp = self.river_temp

    def set_air_temperature(self):
        """
        """
        self.air_temp = self.air_temperature__mean - (
            self.air_temperature__amplitude / 2
        ) * np.cos(np.pi * 2 * self.time / (self.air_temperature__period))

    def animate_riverbank_profile(self, folder="profile_movie/", dt=1):
        for i in range(0, len(self.profiles), dt):
            fig, ax = plt.subplots(1, 1)
            ax.fill_between(
                x=self.profiles[i].x,
                y1=self.profiles[i].z,
                y2=np.ones_like(self.profiles[i].z) * -2,
                color="brown",
            )

            # plt.colorbar(sc, label="surface temperature \n [$\degree$ C]")
            ax.set_title(
                "profile evolution \n DOY = " + str(int(self.profiles[i].age % 365))
            )

            ax.set_xlabel("x (m)")
            ax.axis("equal")
            ax.set_ylim((0, self.bank_height + 1))
            # plt.xlim((8, 12))
            ax.set_ylabel("elevation (m)")
            plt.savefig(
                folder
                + "riverbankprofile"
                + str(int(self.profiles[i].age * 100)).zfill(10)
                + ".png",
                dpi=500,
            )
            plt.close()
        make_animation(folder, moviename="riverbankprofile")

    def plot_riverbank_temp(self, folder="temp_movie/", vmin=-5, vmax=5, show=False):
        imshow_grid(
            self.grid,
            self.grid.at_node["temperature"] - 273.15,
            color_for_closed="blue",
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
        )
        plt.title(
            "river bank temperature ($\degree$ C) \n DOY = " + str(int(self.time % 365))
        )

        if show == True:
            plt.show()
        else:
            plt.savefig(
                folder + "riverbanktemp" + str(int(self.time * 10)).zfill(10) + ".png",
                dpi=500,
            )
            plt.close()


def make_animation(folder, moviename):
    images = []
    for this_name in sorted(os.listdir(folder)):
        if this_name[-3:] == "png":
            # print("appending " + this_name)
            images.append(imageio.imread(folder + this_name))

    imageio.mimsave(moviename + "_movie.gif", images)
    return


"""
under construction
def resample_profile(x, z, s_spacing):
    ds = np.concatenate(
        ([0], np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2)))
    )  # compute derivatives
    # npts = int(ds[-1] / s_spacing)
    new_ds = np.arange(0, 1, s_spacing)
    tck, u = interpolate.splprep([x, z], s=0)
    # unew = np.linspace(0, 1, 1 + int(ds[-1] / s_spacing))  # vector for resampling
    out = interpolate.splev(new_ds, tck)  # resampling
    new_x, new_z = out[0], out[1]

    return new_x, new_z
"""


def get_air_temperature(
    t, mean, amplitude, period,
):

    air_temp = mean - (amplitude / 2) * np.cos(np.pi * 2 * t / (period))
    return air_temp
