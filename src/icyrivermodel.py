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
from scipy.interpolate import UnivariateSpline
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class profile:
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
            representative time - or age - of profile, seconds
        """
        self.x = x
        self.z = z
        self.T = T
        self.age = t


class IcyRiver:
    def __init__(self, params={}):
        # load stage data
        self.load_river_stage_timeseries()

        # unpack parameters

        ### timing
        self.final_time = params["run_duration"]
        self.time = 0
        self.dx = params["dx"]
        self.dt = params["dt"]
        self.save_dt = params["save_dt"]

        ### air temperature
        self.air_temperature__mean = params["air_temperature__mean"]
        self.air_temperature__amplitude = params["air_temperature__amplitude"]
        self.air_temperature__period = params["air_temperature__period"]
        self.air_temp = self.air_temperature__mean - self.air_temperature__amplitude / 2

        ### riverbank geometry
        self.river_stage = self.stage_by_doy[0]
        self.river_half_width = params["river_half_width"]
        self.bank_height = params["bank_height"]
        self.bank_width = params["bank_width"]
        self.riverbed_z = 0
        self.riverbed_temp = params["riverbed_temp"]
        self.morph_factor = params["morph_factor"]

        ##critical slope between notch and ledge
        self.dzdxcrit = 1

        ### subaerial erosion rate constant
        self.alpha_are = params["subaerial_erosion"]

        ### river temperature and seasonality
        self.river_temp = 273.15
        self.river_temp_max = params["water_temperature__max"]
        self.trib = params["trib_doy"]  # 140 doy
        self.tfreeze = params["tfreeze_doy"]  # 310
        self.thighstage = self.trib + 31

        ### material properties
        self.L_ice = 334000  # J/kg
        self.C_ice = 2000  # J/kgK
        self.C_water = 4000  # J/kgK
        self.C_perma = 1000
        self.rho_ice = 917  # kg/m3
        self.k_ice = 2  # J/smK
        self.k_perma = 0.1  # J/smK
        self.rho_perma = 1500  # kg/m3
        self.melt_temp = 273.15  # K
        self.W = params["water_content"] / 100  # .5

        # create landlab grid
        n_rows = (self.bank_height + abs(params["lower_bound__depth"]) + 1) / self.dx
        n_cols = (self.bank_width + self.river_half_width) / self.dx
        self.grid = RasterModelGrid(
            (n_rows, n_cols),
            xy_spacing=self.dx,
            xy_of_lower_left=(0, params["lower_bound__depth"]),
            xy_of_reference=(0, params["lower_bound__depth"]),
        )
        ### add fields for heat diffusion
        self.T = self.grid.add_zeros("temperature", at="node")
        self.qT = self.grid.add_zeros("heat_flux", at="link")

        self.erosion = self.grid.add_zeros("eroded", at="node")
        ### divide core nodes into air, river, riverbed

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
        self.profile_dz = params["profile_dz"]  # m
        self.profile_npts = np.ceil(
            (self.bank_width + self.bank_height + self.river_half_width)
            / self.profile_dz
        )
        x, z = self.create_initial_profile()
        # grab temperature at bnklne nodes
        T, node_ids = self.map_temp_to_bank(x, z)
        # save first profile object
        self.profiles = []
        self.profiles.append(profile(x, z, T, 0))

    def run(self):
        """Run simulation from start to finish"""

        while self.time <= self.final_time:
            # keep time
            self.time += self.dt
            # update bcs
            self.set_air_temperature()
            self.set_river_temperature()
            doy = int((self.time / (60 * 24 * 60)) % 365)
            self.river_stage = self.stage_by_doy[doy] * (
                1 + ((np.random.random(1) - 0.5) * 0.01)
            )

            self.T[self.air_nodes] = self.air_temp
            self.T[self.river_nodes] = self.river_temp
            self.T[self.riverbed_nodes] = self.riverbed_temp
            # calculate temperature on landlab grid
            grad = self.grid.calc_grad_at_link(self.T)
            link_temp = self.grid.map_mean_of_link_nodes_to_link("temperature")
            #    kappa = (
            #            np.ones_like(self.grid.active_links)
            #                * (self.k_ice)#
            #            / (self.rho_ice * self.C_ice)
            #            )
            kappa = (self.k_perma) / (self.rho_perma * self.C_perma)
            self.qT[self.grid.active_links] = kappa * grad[self.grid.active_links]
            dTdt = self.grid.calc_flux_div_at_node(self.qT)
            self.T[self.grid.core_nodes] += dTdt[self.grid.core_nodes] * self.dt

            # map temperature to profile
            x = self.profiles[-1].x
            z = self.profiles[-1].z

            if self.time % self.save_dt == 0:
                self.plot_riverbank_temp()

            (T, node_ids) = self.map_temp_to_bank(x, z)

            # calculate erosion
            if self.time > 0:  # 60 * 60 * 24 * 365:
                E = self.calc_thermal_erosion(x, z, T, dTdt[node_ids])
                E += self.calc_failure(x, z)

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
        new_air = []
        for i in self.river_nodes:
            if self.grid.y_of_node[i] >= self.river_stage:
                # take off riverlist put on air list
                new_air.append(i)
        if len(new_air) > 0:
            self.river_nodes = np.setdiff1d(self.river_nodes, new_air)
            self.air_nodes = np.concatenate((self.air_nodes, new_air))

        new_water = []
        for i in self.air_nodes:
            if self.grid.y_of_node[i] < self.river_stage:
                # take off airlist put on river list
                new_water.append(i)
        if len(new_water) > 0:
            self.air_nodes = np.setdiff1d(self.air_nodes, new_water)
            self.river_nodes = np.concatenate((self.river_nodes, new_water))

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
        bluff_id = np.where((z < self.bank_height) & (z > self.riverbed_z))[0][:]
        flat_id = np.where((z == self.bank_height) & (z == self.riverbed_z))[0][:]

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

    def get_boundary_shear_stress(self, z, dzdx, S=0.005, k=0.4, D84=0.064):
        rhos = self.rho_perma
        g = 9.81
        H = self.river_stage - z
        if dzdx > 0:
            H = (z - self.riverbed_z) / 2

        uz = np.sqrt(g * H * S)
        tb = rhos * (uz * k) ** 2 * (np.log(10 * z / D84) ** -2)
        return tb

    def calc_thermal_erosion(self, x, z, T, dTdt):
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
        # vertical slope
        dzdx = np.concatenate((np.diff(z) / np.diff(x), [0]))
        # nodes below water
        below_water = np.where(
            (z > self.riverbed_z) & (z <= self.river_stage)  # & (T < self.melt_temp)
        )[0][:]

        # list of nodes where ground ice is present and river wtare ihas thawed

        # Latent heat of ice
        erosion = np.zeros_like(x)

        H = np.zeros_like(x)
        H[below_water] = self.river_stage - z[below_water]
        S = 0.0001
        tauc = 1000 * 9.81 * S * 0.05  # self.get_boundary_shear_stress(0.5)

        # T(t) - T(t-1)
        k_d = 0.0000001
        # Frozen erosion

        for i in below_water:
            tau = self.get_boundary_shear_stress(z[i], dzdx[i])

            w = self.W

            fluvial = (
                -k_d
                * (tau - tauc)
                * (1 - np.exp((self.melt_temp - T[i]) / self.melt_temp))
            )
            if fluvial > 0:
                fluvial = 0

                #            T[i] < self.melt_temp:

                # temp_change = self.grid.dx * -T_div[i] / (self.melt_temp - T[i])
            k = self.k_perma

            conduction = -k * (self.river_temp - T[i])
            denom = self.L_ice * self.rho_ice * w

            erosion[i] = (
                self.dt
                * self.morph_factor
                * ((self.dx * dTdt[i] + conduction) / denom + fluvial)
            )

            #    else:
            #    erosion[i] = (
            #            self.dt
            #            * self.morph_factor
            #            * ((fluvial))  # + convection / denom)
            #        )

        # Thawed erosion

        # subaerial melting
        above_water_bot = np.where(z > self.river_stage)[0][-1]
        above_water_top = np.where(z == self.bank_height)[0][-1]

        erosion[above_water_bot:above_water_top] = (
            -self.alpha_are
            * self.dt
            * (self.air_temp - self.T[above_water_bot:above_water_top])
        )
        erosion[erosion > 0] = 0
        # if np.sum(erosion) < 0:
        #    print(erosion[below_water])
        return erosion

    def calc_failure(self, x, z):
        """
        shear off overhang when bank extends past critical slope

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
        # find the node index of the left-most node
        notch = np.where(
            x
            == np.min(x[np.where((z > self.riverbed_z) & (z < self.bank_height))[0][:]])
        )[0][0]

        # right-most point on horizontal surface
        ledge = np.where(z == self.bank_height)[0][-1]
        slopecrit = (z[ledge] - z[notch]) / (x[ledge] - x[notch])
        # print((x[hinge] - x[top]))
        if (slopecrit <= self.dzdxcrit) & (slopecrit > 0):
            # all nodes between ledge and notch
            overhang = np.where((x >= x[notch]) & (z > z[notch]))[0][:]
            # print(len(notch))
            erosion[overhang] = x[notch] - x[overhang]
            erosion[erosion > 0] = 0
            # erosion[erosion > 0] = 0
            print(
                "slope failure at DOY: "
                + str(int((self.time / (24 * 60 * 60)) % (365)))
            )
        return erosion

    def create_grid(self):
        """
        create river bank landlab grid
        """
        n_rows = (self.bank_width + self.river_half_width) / self.dx
        n_cols = (self.bank_height + self.lower_bound__depth) / self.dx
        self.grid = RasterModelGrid((n_rows, n_cols), xy_spacing=self.spacing)

    def create_initial_profile(self, options="rectangle"):
        """
        create profile object based on initial condition parameters
        options are a rectangle or CR22_T5_bank1.

        """
        if options == "rectangle":
            # top

            x_top = np.asarray([0, self.bank_width])

            z_top = np.ones(2) * (self.riverbed_z + self.bank_height)
            # bluff
            z_bluff = np.linspace(
                self.bank_height + self.riverbed_z,
                self.riverbed_z,
                int(self.bank_height / self.profile_dz),
            )

            x_bluff = np.ones_like(z_bluff) * self.bank_width
            # bed
            x_bed = np.arange(
                self.bank_width + self.dx,
                (self.bank_width + self.river_half_width) - self.dx,
                self.profile_dz,
            )
            z_bed = np.ones_like(x_bed) * self.riverbed_z

            x = np.concatenate((x_top, x_bluff, x_bed))
            z = np.concatenate((z_top, z_bluff, z_bed))

            return x, z

        if options == "realistic":
            # top
            x_top = np.linspace(
                0, (self.bank_width), int(self.bank_width / self.profile_dz)
            )
            z_top = np.ones_like(x_top) * (self.riverbed_z + self.bank_height)
            # bluff
            z_bluff = np.linspace(
                self.bank_height + self.riverbed_z,
                self.riverbed_z + self.bank_height / 3,
                int(self.bank_height / self.profile_dz),
            )

            x_bluff = np.ones_like(z_bluff) * x_top[-1] - 0.1 * (
                z_bluff - self.bank_height
            )
            # bed
            x_bed = np.arange(
                x_bluff[-1] + self.dx,
                (self.bank_width + self.river_half_width) - self.dx,
                self.profile_dz * 3,
            )
            z_bed = np.ones_like(x_bed) * z_bluff[-1] - 0.05 * (x_bed - x_bluff[-1])

            x = np.concatenate((x_top, x_bluff, x_bed[z_bed >= -1]))
            z = np.concatenate((z_top, z_bluff, z_bed[z_bed >= -1]))
            plt.plot(x, z)
            plt.show()
            return x, z

    def resample_profile(self, x, z, s_spacing):
        ds = np.concatenate(
            ([0], np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2)))
        )  # compute derivatives
        # npts = int(ds[-1] / s_spacing)
        new_ds = np.arange(0, 1, s_spacing)
        znew = np.arange(
            np.max(z), np.min(z), 100
        )  # np.linspace(np.max(z), np.min(z), -self.profile_dz)
        f = interpolate.interp1d(x, z)
        xnew = f(znew)
        # unew = np.linspace(0, 1, 1 + int(ds[-1] / s_spacing))  # vector for resampling

        return xnew, znew

    def load_river_stage_timeseries(self, filepath="data/staines_depth_at_DOY.csv"):
        df = pd.read_csv(filepath)
        self.stage_by_doy = df.Depth.values

    def set_river_temperature(self):
        """
        """
        doy = int((self.time / (60 * 60 * 24)) % 365)

        if (doy) >= self.trib and (doy) <= self.thighstage:
            self.river_temp = self.melt_temp + (
                self.river_temp_max - self.melt_temp
            ) * ((doy - self.trib) / (self.thighstage - self.trib))
            self.riverbed_temp = self.river_temp
        if (doy) > self.thighstage and (doy) <= self.tfreeze:
            self.river_temp = self.melt_temp + (
                self.river_temp_max - self.melt_temp
            ) * ((self.tfreeze - doy) / ((self.tfreeze - self.thighstage)))

            self.riverbed_temp = self.river_temp
        else:
            self.river_temp = self.melt_temp
            self.riverbed_temp = self.river_temp

    def set_air_temperature(self):
        """
        """
        doy = int((self.time / (60 * 60 * 24)) % 365)

        self.air_temp = self.air_temperature__mean - (
            self.air_temperature__amplitude / 2
        ) * np.cos(np.pi * 2 * doy / (self.air_temperature__period))

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
                "profile evolution \n DOY = "
                + str((self.profiles[i].age / (60 * 60 * 24)) % 365)
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

    def plot_riverbank_temp(self, folder="temp_movie/", vmin=-7, vmax=7, show=False):

        imshow_grid(
            self.grid,
            self.grid.at_node["temperature"] - 273.15,
            color_for_closed="blue",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            allow_colorbar=True,
            colorbar_label="$\degree$ C",
        )
        doy = int((self.time / (60 * 24 * 60)) % 365)
        plt.title("river bank temperature \n DOY = " + str(doy))

        if show == True:
            plt.show()
        else:
            plt.savefig(
                folder + "riverbanktemp" + str(int(self.time)).zfill(10) + ".png",
                dpi=500,
            )
            plt.close()


def make_animation(folder, moviename):
    images = []
    for this_name in sorted(os.listdir(folder)):
        if this_name[-3:] == "png":
            # ("appending " + this_name)
            images.append(imageio.imread(folder + this_name))

    imageio.mimsave(moviename + "_movie.gif", images)
    return


def get_air_temperature(
    t, mean, amplitude, period,
):

    doy = int((t / (60 * 60 * 24)) % 365)
    air_temp = mean - (amplitude / 2) * np.cos(np.pi * 2 * doy / period)
    return air_temp
