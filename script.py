# python3

import os
import sys
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


##################################
#                                #
# Biology HL Internal Assessment #
#            IB 2019             #
#                                #
##################################


"""
############################
# Explanation of this file #
############################

If .csv files are needed, input one (any) sys.argv.
This will save raw and processed data tables,
as well as the t-test table.
This will also give a .png image of the graph.

#####
This file analyses data from Allan Brain Atlas,
"Age and Dementia !!!!"
to answer the research question:
  How does the pTau-181 protein concentration change
  in the hippocampus, the temporal cortex, the frontal
  white matter, and the parietal cortex in the brain
  with the progression of Braak stages?

#####
The following are the steps this programme:
1. reads the csv files (using pandas library)
2. extracts the relevant data (donor code name, Braak
   stage, and pTau-181 concentration for each brain
   structure studied - HIP, TCx, FWM, PCx) (using pandas)
3. calculates means and standard errors for each group (each
   Braak stage for each brain structure) using NumPy library
4. makes the appropriate graphs (using pyplot from
   the Matplotlib library)
5. does a t-test statistical analysis (using the ttest_ind
   function from stats in the SciPy library).


#####
Functions written in this file:
* fn_make_raw_df: makes raw data tables
* fn_stderr: finds standard error
* fn_make_processed_table: makes processed data table
* fn_save_csv: saves tables as csv
* fn_make_graph: makes graph
* fn_t_test: does t_test

#####
All variable names are written in Hungarian notation:
fn = function
i = integer
f = float
b = boolean
s = string
t = tuple
l = list
na = numPy array
df = pandas DataFrame
ser = pandas Series
plt = matplotlib.pyplot plot
fig = matplotlib.pyplot figure
ax = matplotlib.pyplot axes

"""


def fn_make_raw_df(na_donors, na_braak, df_protein_data, s_input_structure):
    """
    Makes a pandas DataFrame for the input structure chosen by collecting
    only relevant data from the original DataFrame provided by the csv file
    "ProteinAndPathologyQuantifications.csv".
    Also removes all nans, and adjusts the donor and Braak information.
    :param na_donors: 1-D array of all donor names (size N = number of donors)
    :param na_braak: 1-D array of size N with Braak stages of each donor
                     in correct order
    :param df_protein_data: DataFrame of full csv file
    :param s_input_structure: the brain structure in question, either:
                              "HIP", "TCx", "FWM", "PCx"
    :return: the selected DataFrame N-row and 3-col: "donor", "braak", "ptau"
    """

    # number of donors
    i_n_donors = len(na_donors)

    # changes name of structure to lower case
    s_input_structure = s_input_structure.lower()

    # structure names
    l_struct_names = ["hip", "tcx", "fwm", "pcx"]
    # checks if structure name is appropriate
    if s_input_structure not in l_struct_names:
        raise ValueError("Unexpected name of structure: should be one of %s" % l_struct_names)

    # initialises the 1-D array
    na_struct = np.full(i_n_donors, np.nan)

    # appends to na_struct where not "nan"
    for i_idx, s_donor in enumerate(na_donors):
        # finds all the regions available for donor
        df_donor_data = df_protein_data[df_protein_data["donor_name"] == s_donor]
        # for each row, meaning brain structure of the particular donor's data
        for _, ser_structure in df_donor_data.iterrows():
            # finds the structure name of row (and converts to lower case)
            s_structure = ser_structure["structure_acronym"].lower()
            # updates the information into ndarray if correct structure
            if s_structure == s_input_structure:
                na_struct[i_idx] = ser_structure["ptau_ng_per_mg"]

    # finds indices where there are nans in na_struct
    na_nan_idx = np.isnan(na_struct)

    # finds number of nans
    i_nan_num = sum(na_nan_idx)

    # finds indices where NOT nans in na_struct
    na_nonnan_idx = ~na_nan_idx

    # adjusts all ndarrays to remove data with nans
    na_donors = na_donors[na_nonnan_idx]
    na_braak  = na_braak[na_nonnan_idx]
    na_struct = na_struct[na_nonnan_idx]

    # makes M-row x 3-col ndarray for structure data,
    # M is number of donors - number of nan values
    na_struct_full = np.column_stack((na_donors, na_braak, na_struct))

    # checks that the rownum is correct
    if na_struct_full.shape[0] + i_nan_num != i_n_donors:
        raise ValueError("Size of matrix for %s structure is incorrect: problem with nans"
                         % s_input_structure)

    # column names
    t_col_names = ("donor", "braak", "ptau")

    # make DataFrames using the ndarray made
    df_struct = pd.DataFrame(data=na_struct_full, columns=t_col_names)

    return df_struct
#


def fn_stderr(na_stddev, na_observations_num):
    """
    Calculates standard of error
    :param na_stddev: 1-D N-length array of std deviations
    :param na_observations_num:  1-D N-length array of number of observations
    :return: ndarray of standard error
    """
    return na_stddev / np.array([sqrt(i) for i in na_observations_num])
#


def fn_into_braaks(df_struct, s_type):
    """
    If s_type == "all":
    Returns a list of 1-D ndarrays N0, N1, ..., N5, N6,
    where the length of each ndarray corresponds to the number
    of donors at each Braak stage.
    If s_type == "combined":
    Returns a list of 1-D ndarrays N0, N(1,2), N(3,4), N(5,6),
    where the length of each ndarray corresponds to the number
    of donors at each set of Braak stages.
    :param df_struct: raw data frame of particular brain structure
    :param s_type: either "all" = stages range(7)
                       or "combined" = stages combined in groups
                           ((0), (1, 2), (3, 4), (5, 6))
    :return: depends; explained above
    """

    # separate cases for s_type
    if s_type == "all":
        # list of separate lists for each Braak stage
        l_into_braaks = [np.array(df_struct[df_struct["braak"] == i]["ptau"]) for i in range(7)]

    elif s_type == "combined":
        # (note: excludes Braak stage 0)
        l_stages = [[1, 2], [3, 4], [5, 6]]
        # first, l_into_braaks is initiated
        # with results for Braak stage 0 already appended
        l_into_braaks = [np.array(df_struct[df_struct["braak"] == 0]["ptau"])]
        for l_stg in l_stages:
            # a whole array with data from both stages of combined group
            na_stage = np.array(df_struct[df_struct["braak"] == l_stg[0]]["ptau"].append(
                       df_struct[df_struct["braak"] == l_stg[1]]["ptau"]))
            l_into_braaks.append(na_stage)

    else:
        raise ValueError('Unexpected s_type passed as argument: should be "all" or "combined"')

    return l_into_braaks
#


def fn_make_processed_table(df_struct, s_type):
    """
    Makes the processed data table
    4 columns: mean, stddev, observation num, stderr of ptau-181
    Rows are Braak stages, determined by s_type
    :param df_struct: raw df from fn_make_raw_df of particular structure
    :param s_type: either "all" = stages range(7)
                       or "combined" = stages combined in groups
                           ((0), (1, 2), (3, 4), (5, 6))
    :return: a B-row x 4-col ndarray, where B is the number of Braak stages,
             depending on s_type
    """

    # separates into Braak stages
    l_into_braaks = fn_into_braaks(df_struct, s_type)

    # makes a ndarray of the means, standard deviations, observation num, standard errors
    na_mean = np.array([np.mean(i) for i in l_into_braaks])
    na_stddev = np.array([np.std(i) for i in l_into_braaks])
    na_observations_num = np.array([len(i) for i in l_into_braaks])
    na_stderr = fn_stderr(na_stddev, na_observations_num)

    # makes a 2-D matrix
    na_matrix = np.column_stack((na_mean, na_stddev, na_observations_num, na_stderr))

    return na_matrix
#


def fn_save_csv(na_matrix, s_struct, s_type, s_path):
    """
    (Primarily) for saving fn_make_processed_table data
    (this function is only used for fn_make_processed_table)
    :param na_matrix: processed data matrix
                      (returned by fn_make_processed_table)
    :param s_struct: the brain structure
    :param s_type: either "all" or "combined"
    :param s_path: path to the folder in which to save csv
    :return: no return - save directly
    """

    # makes the file nameT
    s_filename = f"ptau_Processed_{s_struct}_{s_type}.csv"

    # makes dataframe, and uses pandas to save easily
    df_save = pd.DataFrame(na_matrix)
    df_save.to_csv(os.path.join(s_path, s_filename))
#


def fn_make_graph(na_hip, na_tcx, na_fwm, na_pcx, s_type, s_path_out=""):
    """
    Makes bar charts with Braak stage as x-axis, pTau-181/Tau
    ratio at y-axis
    N = number of donors
    Can make .png image of graph if s_path_out is included
    :param na_hip: N-row x 3-col with data for HIP
    :param na_tcx:             "               TCx
    :param na_fwm:             "               FWM
    :param na_pcx:             "               PCx
    :param s_type: can be either "all" or "combined"
                   "all" gives graph with all Braak stages separate
                   "combined" gives graph with stages (0), (1, 2),
                   (3, 4), (5, 6) which are combined
    :param s_path_out: name of path to output
    :return: no return - plots graph
    """

    # check s_type
    if s_type != "all" and s_type != "combined":
        raise ValueError("s_type must be either: %s or %s" % ("all", "combined"))

    # check that s_type is consistent with the input data
    if s_type == "all" and ([len(na_hip), len(na_tcx), len(na_fwm), len(na_pcx)] != [7] * 4):
        raise ValueError("fn_make_graph: s_type inconsistent with data inputted")
    if s_type == "combined" and ([len(na_hip), len(na_tcx), len(na_fwm), len(na_pcx)] != [4] * 4):
        raise ValueError("fn_make_graph: s_type inconsistent with data inputted")

    # legend series - brain structure names
    t_regions = ("HIP", "TCx", "FWM", "PCx")

    # initialise plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # customises xticks according to s_type
    if s_type == "all":
        # number of independent variables
        i_n_indvars = 7
        na_xticks = np.arange(-0.1, i_n_indvars - 0.1, 1)
        plt.xticks(na_xticks, ["0", "I", "II", "III", "IV", "V", "VI"])

    elif s_type == "combined":
        # number of independent variables
        i_n_indvars = 4
        na_xticks = np.arange(-0.1, i_n_indvars - 0.1, 1)
        plt.xticks(na_xticks, ["0", "I and II", "III and IV", "V and VI"])

    # graph aesthetics
    na_ind = np.arange(i_n_indvars)
    f_width = 0.2

    # plots bars for each brain structure
    na_hip_vals = na_hip[:, 0]
    na_hip_err  = na_hip[:, 3]
    plt_hip = ax.bar(na_ind-f_width*2, na_hip_vals, f_width, color="b",  yerr=na_hip_err, capsize=5)

    na_tcx_vals = na_tcx[:, 0]
    na_tcx_err  = na_tcx[:, 3]
    plt_tcx = ax.bar(na_ind-f_width,   na_tcx_vals, f_width, color="C2", yerr=na_tcx_err, capsize=5)

    na_fwm_vals = na_fwm[:, 0]
    na_fwm_err  = na_fwm[:, 3]
    plt_fwm = ax.bar(na_ind,           na_fwm_vals, f_width, color="r",  yerr=na_fwm_err, capsize=5)

    na_pcx_vals = na_pcx[:, 0]
    na_pcx_err  = na_pcx[:, 3]
    plt_pcx = ax.bar(na_ind+f_width,   na_pcx_vals, f_width, color="C1", yerr=na_pcx_err, capsize=5)

    # graph cleanup - title, labels, legend
    plt.title(            "The average pTau-181 protein concentration \n \
               in different brain structures with the progression of Braak stages", fontsize=16)
    plt.xlabel("Braak Stage")
    plt.ylabel("pTau-181 concentration (ng / mg)")

    ax.legend((plt_hip[0], plt_tcx[0], plt_fwm[0], plt_pcx[0]), t_regions, loc=2, fontsize="medium", title="Brain Structures")

    plt.show()

    if s_path_out:
        fig.savefig(os.path.join(s_path_out, "graph.png"))
#


def fn_t_test(na1, na2):
    """
    Uses scipy.stats builtin t-test
    :param na1, na2: two numpy vector arrays;
                     can be different size
    :return: tuple of (t-statistic, p-value)
    t-statistic is made positive by abs value
    """
    # finds t-statistic and p-value
    i_tstat, i_pval = ttest_ind(na1, na2)

    return (abs(i_tstat), i_pval)
#


def fn_t_test_struct(df_struct, s_type):
    """
    Does t-test for one brain structure
    :param df_struct: raw df from fn_make_raw_df of particular structure
    :param s_type: either "all" = stages range(7)
                       or "combined" = stages combined in groups
                           ((0), (1, 2), (3, 4), (5, 6))
    :return: ndarray size (1, 6)
    """

    # separates into Braak stages
    l_into_braaks = fn_into_braaks(df_struct, s_type)

    # makes table structure
    #  Between Braak 0&(I, II); (I, II)&(III, IV); (III, IV)&(V, VI)
    #  |   t-stat | p-val     |  t-stat | p-val  |  t-stat | p-val |
    # "|" indicates separate values of array
    l_ttable = []

    for i in range(3):
        t_ttest = fn_t_test(l_into_braaks[i], l_into_braaks[i+1])
        l_ttable.append(t_ttest[0])
        l_ttable.append(t_ttest[1])

    return np.array(l_ttable)
#


def fn_t_test_table(df_hip, df_tcx, df_fwm, df_pcx, s_type):
    """
    Uses fn_t_test_struct function to make t_test table

         | Between Braak 0&(I, II) | (I, II)&(III, IV) | (III, IV)&(V, VI) |
    -----|-------------------------|-------------------|-------------------|
     HIP |     t-stat | p-val      |   t-stat | p-val  |   t-stat | p-val  |
     TCx |     t-stat | p-val      |   t-stat | p-val  |   t-stat | p-val  |
     FWM |     t-stat | p-val      |   t-stat | p-val  |   t-stat | p-val  |
     PCx |     t-stat | p-val      |   t-stat | p-val  |   t-stat | p-val  |

    :param df_hip, df_tcx, df_fwm, df_pcx: raw df from
               fn_make_raw_df of each brain structure
    :param s_type: either "all" = stages range(7)
                       or "combined" = stages combined in groups
                           ((0), (1, 2), (3, 4), (5, 6))
    :return: ndarray table size (4, 6), as shown above
    """

    # get t-test statistics for each brain structure
    na_t_hip = fn_t_test_struct(df_hip, s_type)
    na_t_tcx = fn_t_test_struct(df_tcx, s_type)
    na_t_fwm = fn_t_test_struct(df_fwm, s_type)
    na_t_pcx = fn_t_test_struct(df_pcx, s_type)

    # combine them
    na_ttable = np.vstack((na_t_hip, na_t_tcx, na_t_fwm, na_t_pcx))

    return na_ttable
#


if __name__ == '__main__':

    # path to files TODO change this
    s_path_in = "/Users/ ... path to input data ... "
    s_path_out = "/Users/ ... path to output data ... "

    # makes names of two files, combined with their appropriate path
    s_donor_info_fname = os.path.join(s_path_in, "DonorInformation.csv")
    s_protein_fname = os.path.join(s_path_in, "ProteinAndPathologyQuantifications.csv")

    # reads two data files into pandas Dataframe
    # pd_info contains: donor code name, Braak stage, dementia or no dementia
    df_info = pd.read_csv(s_donor_info_fname)
    # pd_protein_data contains: donor code name, brain structures, protein concentrations
    #                           where I will get pTau-181 and Tau ratios
    df_protein_data = pd.read_csv(s_protein_fname)

    # make my own DataFrame with relevant information selected
    # 4 DataFrames will be made for each structure of the brain:
    # HIP, TCx, FWM, PCx

    # start off with ndarrays
    na_donors = np.array(df_info["name"])
    na_braak = np.array(df_info["braak"])

    # make DataFrames for each structure
    df_hip = fn_make_raw_df(na_donors, na_braak, df_protein_data, "HIP")
    df_tcx = fn_make_raw_df(na_donors, na_braak, df_protein_data, "TCx")
    df_fwm = fn_make_raw_df(na_donors, na_braak, df_protein_data, "FWM")
    df_pcx = fn_make_raw_df(na_donors, na_braak, df_protein_data, "PCx")

    # save raw data as .csv files if sys argument exists
    if len(sys.argv) > 1:
        df_hip.to_csv(os.path.join(s_path_out, "raw_HIP.csv"))
        df_tcx.to_csv(os.path.join(s_path_out, "raw_TCx.csv"))
        df_fwm.to_csv(os.path.join(s_path_out, "raw_FWM.csv"))
        df_pcx.to_csv(os.path.join(s_path_out, "raw_PCx.csv"))

    # parameter s_type (detailed in fn_make_processed_table
    s_type = "combined"  # "all"

    # make processed data ndarray matrix for each structure
    na_hip = fn_make_processed_table(df_hip, s_type)
    na_tcx = fn_make_processed_table(df_tcx, s_type)
    na_fwm = fn_make_processed_table(df_fwm, s_type)
    na_pcx = fn_make_processed_table(df_pcx, s_type)

    # save as .csv file if sys argument exists
    if len(sys.argv) > 1:
        fn_save_csv(na_hip, "HIP", s_type, s_path_out)
        fn_save_csv(na_tcx, "TCx", s_type, s_path_out)
        fn_save_csv(na_fwm, "FWM", s_type, s_path_out)
        fn_save_csv(na_pcx, "PCx", s_type, s_path_out)

    # make the bar chart, make .png image if s_path_out is included
    if len(sys.argv) > 1:
        fn_make_graph(na_hip, na_tcx, na_fwm, na_pcx, s_type, s_path_out)
    else:
        fn_make_graph(na_hip, na_tcx, na_fwm, na_pcx, s_type)

    # t-test table. save as .csv file if sys argument exists
    if len(sys.argv) > 1:
        na_ttest = fn_t_test_table(df_hip, df_tcx, df_fwm, df_pcx, s_type)
        df_save = pd.DataFrame(na_ttest)
        df_save.to_csv(os.path.join(s_path_out, "ttest.csv"))
#
