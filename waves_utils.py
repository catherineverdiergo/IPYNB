# ----------------------------------------
# -*- coding: utf-8 -*-
# created by Catherine Verdier on 29/09/2016
# ----------------------------------------

import pandas as pd
import aphp_waves_dic as wd
from matplotlib import pyplot as plt
from datetime import timedelta
import numpy as np
import drill_queries as dq


def to_time_series(df_wave):
    """
    Transform a pandas wave dataframe to a time series
    :param df_wave:
    :return: None
    """
    df_wave.drop('id_measure_type', axis=1, inplace=True)
    df_wave.index = df_wave['dt_insert']
    df_wave.drop('dt_insert', axis=1, inplace=True)


def add_time_ticks(t_serie, dt_min, dt_max):
    """
    Add time ticks to a time series
    :param t_serie: time series to update
    :param dt_min: time tick to insert at the begining (with a None value)
    :param dt_max: time tick to insert at the end (with a None value)
    :return: the updated time series
    """
    stop_serie = pd.Series(index=[dt_max])
    start_serie = pd.Series(index=[dt_min])
    t_serie = pd.concat([start_serie, t_serie, stop_serie])
    t_serie.drop(0, axis=1, inplace=True)
    return t_serie


def resample_and_interpolate(t_serie, delay, nb_pts, method='time', order=3):
    """
    Resample and interpolate a time serie in order to get exactly nb_pts points
    :param t_serie: time series to resample and interpolate
    :param delay: delay between 2 points in minutes
    :param nb_pts: number of points to get at output
    :param method: interpolation method as described at
            http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.interpolate.html
    :param order: interpolation order for methods for which this parameter is needed (as 'polynomial' or 'spline')
    :return: the updated time series
    """
    sp_str = "{}T".format(int(delay))
    t_serie = t_serie.resample(sp_str).mean()
    if len(t_serie) < nb_pts:
        while len(t_serie) < nb_pts:
            dt_max = t_serie.index[len(t_serie) - 1]
            dt_max = dt_max + timedelta(seconds=1)
            t_serie = pd.concat([t_serie, pd.Series(index=[dt_max])])
            t_serie.drop(0, axis=1, inplace=True)
    elif len(t_serie) > nb_pts:
        while len(t_serie) > nb_pts:
            t_serie = t_serie.drop([t_serie.index[len(t_serie) - 1]])
    t_serie = t_serie.interpolate(method=method, order=order, limit=30)
    t_serie = t_serie.ffill()
    t_serie = t_serie.bfill()
    return t_serie


class WavesUtilities:
    """
    Class holding method utilities to process waves
    """

    def __init__(self, voi, spark_context, hdfs_db_path):
        """
        Constructor
        :param voi: list of interest for waves
        :param spark_context: spark context to be able to write Parquet tables
        :param hdfs_db_path: target hdfs directory for Parquet tables
        """
        self._voi = voi
        self._dic = wd.WavesDic()
        self._sc = spark_context
        self._db_dir = hdfs_db_path
        self._dq = dq.DrillQueries("drill_eds", voi)
	self._colors = ['#40bf80', '#668cff', '#ffa64d', '#ff33bb', '#330033', '#4dffc3', '#805500', '#999900']

    def get_waves_from_pdf(self, all_waves):
        """
        Separate measures by wave type regarding to the waves dictionary
        :param all_waves: all measures related to VOI as defined in the waves dictionary selected for a given
                list of case ids
        :return: a dictionary of time series (one entry per VOI)
        """
        result = {}
        for v in self._voi:
            aphp_codes = self._dic.get_all_voi_codes([v])
            df_filter = [False] * len(all_waves)
            for code in aphp_codes:
                df_filter |= all_waves['id_measure_type'] == code
            df_filtered = all_waves[df_filter]
            result[v] = df_filtered.sort_values(by='dt_insert', ascending=1)
        return result

    def plot_waves(self, waves):
        """
        Plot waves in a matplotlib figure
        :param waves: all waves for a given case as a dictionary of time series
        :return: None
        """
        plt.figure(figsize=(15, 5))
        for i, metric in enumerate(waves.keys()):
            label = self._dic.get_label(metric)
            plt.plot(waves[metric].dt_insert.values, waves[metric].value_numeric.values, label=label, color=self._colors[i % len(self._colors)], linewidth=3.0)
            plt.plot(waves[metric].dt_insert.values, waves[metric].value_numeric.values, '*', color='#000000')
            plt.legend()

    def norm_waves(self, df_ranges, df_sensor_case, nb_pts, method='time'):
        """
        Generate a set of matrices for a set of cases (batch processing)
        :param df_ranges: considered cases ids and features (util ranges for measures, death date)
        :param df_sensor_case: dataframe of measures related with case
        :param nb_pts: number of desired points
        :return: a set of matrices (waves for cases) + a set of target (booleans : dead / not dead)
        """
        resultT = np.zeros((len(self._voi), nb_pts), dtype=np.float)
        # separate waves
        waves_dic = self.get_waves_from_pdf(df_sensor_case)
        df_dt_range = df_ranges[(df_ranges['id_nda'] == str(df_sensor_case['id_nda'].unique()[0])) & \
                                (df_ranges['j1'] == str(df_sensor_case['j1'].unique()[0]))]
        # Estimate the interval in minutes to resample
        mn_interval = np.round(((pd.to_datetime(df_dt_range['dt_max']) - \
				pd.to_datetime(df_dt_range['dt_min'])).astype('timedelta64[ms]').astype(int) \
                                / 1e+03 / nb_pts / 60).unique()[0])
        for i, key in enumerate(waves_dic.keys()):
            # add dt_min and dt_max to each time serie
	    serie = waves_dic[key][['dt_insert','value_numeric']]
	    serie.index = serie['dt_insert']
	    serie.drop('dt_insert', axis=1, inplace=True)
            serie = add_time_ticks(serie, df_dt_range['dt_min'], df_dt_range['dt_max'])
            # resample and interpolate
            serie = resample_and_interpolate(serie, mn_interval, nb_pts, method=method)
	    serie['dt_insert'] = serie.index
	    waves_dic[key] = serie
            resultT[i] = serie['value_numeric']
        return resultT, df_dt_range['dt_deces'].unique()[0] != "NaT", waves_dic   # False when survivor / True when dead

    def build_pyriemann_input_matrix(self, nb_splits, nb_pts):
        """
        Build the 3d pyriemann input matrix by blocks with selected waves
        :param nb_splits: number of blocks of ids to create
        :param nb_pts: number of points desired to resample waves
        :return: 3d matrix + target vector (mortality)
        """
        from pyspark.sql.functions import concat
        dfs_ids = self._sc._sql.read.parquet(self._db_dir+"/nda_j1_deces") \
            .select(concat('id_nda', 'j1').alias('id_case')).distinct()
        # split dfs_ids in nb_splits RDDs
        rdd_ids_blocks = dfs_ids.rdd.randomSplit([nb_splits] * nb_splits, 42)
        matrix3d = []
        target_vector = []
        # get dates intervals by case
        df_ranges = self._dq.get_dt_ranges()
        # Process by blocks
        for i, rdd in enumerate(rdd_ids_blocks):
            print ("block {} / {}".format(i + 1, len(rdd_ids_blocks)))
            # convert to Dataframe
            df_block = rdd.toDF()
            # save as temporary parquet file
            df_block.write.parquet(self._db_dir+"/tmp_nda_j1_ids", mode='overwrite')
            # prepare Drill query
            dq = self._dq.queries["QUERY_SELECT_SENSOR_BLOCK"]
            dfp = self._dq.drill_conn.df_from_query(dq)
            keys = dfp['id_case'].unique()
            for key in keys:
                # restriction on id_case
                dfp_key = dfp[dfp['id_case'] == key]
                # separe and normalize several waves by type
                # (Heart Rate / Respiration rate / ABP systolic / ABP diastolic)
                xy4case = self.norm_waves(df_ranges, dfp, nb_pts)
                matrix3d.append(xy4case[0])
                target_vector.append(xy4case[1])
            print("{} keys processed".format(np.size(keys)))
        return np.stack(matrix3d), target_vector

    def build_pyriemann_input_matrix_one_shot(self, nb_pts):
        """
        Build the 3d pyriemann input matrix by blocks with selected waves
        :param nb_pts: number of points desired to resample waves
        :return: 3d matrix + target vector (mortality)
        """
        matrix3d = []
        target_vector = []
        # get dates intervals by case
        df_ranges = self._dq.get_dt_ranges()
        # prepare Drill query
        dq = self._dq.queries["QUERY_SELECT_SENSOR"]
	print("loading rows from db...")
        dfp = self._dq.drill_conn.df_from_query(dq)
	print("done")
        keys = dfp['id_case'].unique()
        for i, key in enumerate(keys):
            # restriction on id_case
            dfp_key = dfp[dfp['id_case'] == key]
            # separe and normalize several waves by type
            # (Heart Rate / Respiration rate / ABP systolic / ABP diastolic)
            xy4case = self.norm_waves(df_ranges, dfp_key, nb_pts)
            matrix3d.append(xy4case[0])
            target_vector.append(xy4case[1])
	    if i % 300 == 0:
                print("{} keys processed".format(i+1))
        return np.stack(matrix3d), target_vector
