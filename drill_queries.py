# ----------------------------------------
# -*- coding: utf-8 -*-
# created by Catherine Verdier on 26/09/2016
# ----------------------------------------

import drill_utilities as du
import aphp_waves_dic as wd
import pandas as pd
import numpy as np
from datetime import timedelta
import pyodbc


class DrillQueries:
    """
    Class to build queries and DataFrames to:
        - get the most frequent types of measures provided by ICUs;
        - filter waves rows per relevant cases;
        - ...
    Aim is to provide descriptive statistics about the bag of measures available
    """

    @property
    def voi(self):
        return self._voi

    @property
    def queries(self):
        return self._queries

    @property
    def drill_conn(self):
        return self._drill_conn

    def __init__(self, dsn, voi):
        """
        Constructor ==> initialize Drill connection and build query templates
        :param dsn: ODBC Drill Data Source Name
        :param voi: List of variables of interest
        """
        self._drill_conn = du.DrillODBC(dsn)
        self._dic = wd.WavesDic()
        self._voi = voi
        self._queries = {
            # get number of measures by type
            'QUERY_MEASURES_COUNTERS': "select count(1) as counter,s.id_measure_type, r.label \
                                        from icu_sensor_24 s, ref_measure r \
                                        where s.id_measure_type = cast(r.code as INT) \
                                        and s.dt_cancel = '' \
                                        group by s.id_measure_type, r.label \
                                        order by counter desc, s.id_measure_type, r.label",
            # Count available cases for the selected VOI
            'QUERY_COUNT_CASES': "select count(distinct CONCAT(cast(id_nda as VARCHAR),cast(TO_DATE(dt_deb) as VARCHAR))) \
                                    from icu_sensor_24 s \
                                    where s.id_measure_type in {} and s.dt_cancel = ''",
            # Count available measures by case for selected VOI
            'QUERY_GET_MEASURES': "select CONCAT(cast(id_nda as VARCHAR),cast(TO_DATE(dt_deb) as VARCHAR)) as id_ndaj1, \
                                    s.id_measure_type, count(1) as counter \
                                    from icu_sensor_24 s \
                                    where s.id_measure_type in {} \
                                    and s.dt_cancel = '' \
                                    group by CONCAT(cast(id_nda as VARCHAR),cast(TO_DATE(dt_deb) as VARCHAR)), s.id_measure_type \
                                    order by CONCAT(cast(id_nda as VARCHAR),cast(TO_DATE(dt_deb) as VARCHAR)), s.id_measure_type",
            # Get patient data by case: age + death date
            'QUERY_CASE_DATA': "select distinct CONCAT(cast(s.id_nda as VARCHAR),cast(TO_DATE(s.dt_deb) as VARCHAR)) as id_ndaj1, \
                                    p.age, p.dt_deces from icu_pat_info p, icu_sensor_24 s \
                                    where p.id_nda = cast(s.id_nda as VARCHAR) \
                                    and s.dt_cancel = ''",
            # Get minimum dates per VOI and per case
            'QUERY_DT_MIN_ICU': "select CONCAT(cast(id_nda as VARCHAR),cast(TO_DATE(dt_deb) as VARCHAR)) as id_ndaj1, \
                                id_measure_type, min(dt_insert) as min_dt \
                                from icu_sensor_24 \
                                where id_measure_type in {} \
                                and dt_cancel = '' \
                                group by CONCAT(cast(id_nda as VARCHAR),cast(TO_DATE(dt_deb) as VARCHAR)), id_measure_type \
                                order by CONCAT(cast(id_nda as VARCHAR),cast(TO_DATE(dt_deb) as VARCHAR)), id_measure_type",
            # Get maximum dates per VOI and per case
            'QUERY_DT_MAX_ICU': "select CONCAT(cast(id_nda as VARCHAR),cast(TO_DATE(dt_deb) as VARCHAR)) as id_ndaj1, \
                            id_measure_type, max(dt_insert) as max_dt \
                            from icu_sensor_24 \
                            where id_measure_type in {} \
                            and dt_cancel = '' \
                            group by CONCAT(cast(id_nda as VARCHAR),cast(TO_DATE(dt_deb) as VARCHAR)), id_measure_type \
                            order by CONCAT(cast(id_nda as VARCHAR),cast(TO_DATE(dt_deb) as VARCHAR)), id_measure_type",
            # Test if a given table exists
            'QUERY_TABLE_EXISTS': "select count(1) from {} limit1",
            # Drop table
            'QUERY_DROP_TABLE': "drop table {}",
            # Create table icu_nda_mouv_ufr_tr (movements related with an ICU unit)
            'CREATE_ICU_MOV': "create table icu_nda_mouv_ufr_tr as \
                                select n.* from nda_mouv_ufr_tr n, icu_ufr u \
                                where u.ids_ufr=n.ids_ufr",
            # Create table icu_sensor_24 (measures done during the first 24h in a ICU unit)
            'CREATE_ICU_SENSOR_24': "create table icu_sensor_24 as \
                                    select TO_DATE(n.dt_deb_mouv_ufr) as dt_deb, s.* from sensors s, \
                                    icu_nda_mouv_ufr_tr n \
                                    where cast(s.id_nda as VARCHAR)=n.id_nda and \
                                    s.dt_insert >= n.dt_deb_mouv_ufr and \
                                    s.dt_insert <= n.dt_deb_mouv_ufr + interval '1' DAY(2)",
            # Create table icu_pat_info
            'CREATE_ICU_PAT_INFO': "create table icu_pat_info as \
                                    select n.id_nda, p.age, \
                                    min(n.dt_deb_mouv_ufr) as min_dt_deb_mouv, \
                                    max(n.dt_deb_mouv_ufr) as max_dt_deb_mouv, \
                                    max(n.dt_fin_mouv_ufr) as max_dt_fin_mouv, \
                                    p.cd_sex_tr, p.dt_deces from icu_nda_mouv_ufr_tr n, patient_tr p \
                                    where p.ids_pat = n.ids_pat and \
                                    n.id_nda in (select distinct cast(id_nda as VARCHAR) from icu_sensor_24) \
                                    group by n.id_nda, p.age, p.cd_sex_tr, p.dt_deces",
            # Create table nda_j1_dt_range
            'CREATE_DT_RANGE': "select n.id_nda, n.j1, min(s.dt_insert) as dt_min, max(s.dt_insert) as dt_max \
                                from nda_j1_deces n, icu_sensor_util s \
                                where CAST(n.id_nda as VARCHAR) = CAST(s.id_nda as VARCHAR) \
                                and CAST(n.j1 as VARCHAR) = CAST(s.dt_deb as VARCHAR) \
                                group by n.id_nda, n.j1 \
                                order by n.id_nda, n.j1",
            # create table of useful sensors
            'CREATE_SENSOR_UTIL': "create table icu_sensor_util as \
                                    select * from icu_sensor_24 where dt_cancel = '' \
                                    and CONCAT(cast(id_nda as VARCHAR),cast(TO_DATE(dt_deb) as VARCHAR)) in \
                                    (select distinct(CONCAT(cast(id_nda as VARCHAR),j1)) from nda_j1_deces)",
            # select sensors related with a temporary table of case ids stored in table tmp_nda_j1_ids
            'QUERY_SELECT_SENSOR_BLOCK': "select id_nda, CAST(dt_deb as VARCHAR) as j1, \
                        CONCAT(CAST(id_nda as VARCHAR), CAST(dt_deb as VARCHAR)) as id_case, \
                        id_measure_type, dt_insert, value_numeric \
                        from icu_sensor_util \
                        where CONCAT(CAST(id_nda as VARCHAR), CAST(dt_deb as VARCHAR)) in \
                        (select id_case from tmp_nda_j1_ids) \
                        order by id_nda, CAST(dt_deb as VARCHAR), id_measure_type, dt_insert",
            # select sensors related with a all cases (no batch)
            'QUERY_SELECT_SENSOR': "select id_nda, CAST(dt_deb as VARCHAR) as j1, \
                        CONCAT(CAST(id_nda as VARCHAR), CAST(dt_deb as VARCHAR)) as id_case, \
                        id_measure_type, dt_insert, value_numeric \
                        from icu_sensor_util \
                        where CONCAT(CAST(id_nda as VARCHAR), CAST(dt_deb as VARCHAR)) in \
                        (select CONCAT(CAST(id_nda as VARCHAR), CAST(j1 as VARCHAR)) from nda_j1_deces) \
                        order by id_nda, CAST(dt_deb as VARCHAR), id_measure_type, dt_insert",
            # Store duration intervals and death date
            'QUERY_DT_RANGES': "select t1.id_nda, t1.j1, t1.dt_min, t1.dt_max, t2.dt_deces \
                                from nda_j1_dt_range t1, nda_j1_deces t2 \
                                where t1.id_nda = t2.id_nda and t1.j1 = t2.j1",
	    # select all cases
	    'QUERY_SELECT_ALL_CASES': "select distinct CONCAT(CAST(id_nda as VARCHAR), CAST(j1 as VARCHAR)) \
				from nda_j1_deces"
        }

    def get_measures_counters(self):
        """
        Returns a pandas DataFrame holding number of measures by type
        :return: a pandas DataFrame
        """
        q = self._queries['QUERY_COUNT_CASES']
        return self._drill_conn.df_from_query(q)

    def get_total_cases(self):
        """
        Returns a pandas DataFrames giving the total number of cases holding at least one of the VOI
        :return: 1 row DataFrame
        """
        q = self._queries['QUERY_COUNT_CASES'].format(wd.np_array_2_string(self._dic.get_all_voi_codes(self._voi)))
        return self._drill_conn.df_from_query(q)

    def get_counter_matrix(self):
        """
        Returns a matrix of counters per case and per VOI
        :return: a DataFrame of row counters per case and VOI
        """
        q = self._queries['QUERY_GET_MEASURES'].format(wd.np_array_2_string(self._dic.get_all_voi_codes(self._voi)))
        grouped_df = self._drill_conn.df_from_query(q)
        return grouped_df.pivot(index='id_ndaj1', columns='id_measure_type', values='counter')

    def case_first_level_filter(self, counter_matrix):
        """"
        First level filter to apply to the counter matrix computed by get_counter_matrix
        We have to check to have at least one measure per case for each VOI except for BT (body temperature)
        If we do not have BT we can:
        * get it from the SAPSII form (it is included in SAPSII computation) if it exists
        * suppose it is normal i.e. ~ 37°C
        :param counter_matrix: matrix of counters per case and per VOI generated by get_counter_matrix
                            (pandas DataFrame)
        :return: filtered counter matrix
        """
        result = counter_matrix
        for v in self._voi:
            if v != 'BT':
                aphp_codes = self._dic.get_all_voi_codes([v])
                bcode = [False] * len(result)
                for code in aphp_codes:
                    bcode |= result[code].notnull()
                result = result[bcode]
        return result

    def case_second_level_filter(self, counter_matrix, min_measures=10):
        """
        Second level filter to apply to the counter matrix computed by get_counter_matrix and filtered
        by case_first_level_filter
        We have to check to have at least min_measures per case for each VOI except for BT (body temperature)
        :param counter_matrix: matrix of counters per case and per VOI generated by get_counter_matrix
                            (pandas DataFrame)
        :param min_measures: minimum number of measures we should have for each VOI to consider the case as
        enough relevant
        :return: filtered counter matrix
        """
        result = counter_matrix
        result = result.fillna(0)
        for v in self._voi:
            if v != 'BT':
                aphp_codes = self._dic.get_all_voi_codes([v])
                counter_v = pd.Series(np.zeros(len(result), dtype=int))
                counter_v.index = result.index
                for code in aphp_codes:
                    counter_v += result[code]
                result = result[counter_v > min_measures]
        return result

    def get_cases_data(self):
        """
        Build a DataFrame with available demographic data per case (age + death date)
        :return: pandas dataframe
        """
        q = self._queries['QUERY_CASE_DATA']
        result = self._drill_conn.df_from_query(q)
        result.index = result['id_ndaj1']
        # print(result.head())
        result.drop('id_ndaj1', axis=1, inplace=True)
        return result

    @staticmethod
    def case_more_age_filter(counter_matrix, demographic_df, age_min=18):
        """
        Filter counter matrix: remove cases having less than 18
        :param counter_matrix: matrix of counters per case and per VOI generated by get_counter_matrix
                            (pandas DataFrame)
        :param demographic_df: dataframe built with get_cases_data method
        :return: filtered counter matrix with related demographic data
        """
        # join the 2 matrix by id
        counter_matrix['id_ndaj1'] = counter_matrix.index
        counter_matrix.drop('id_ndaj1', axis=1, inplace=True)
        return pd.merge(counter_matrix, demographic_df[demographic_df['age'] >= age_min],
                        left_index=True, right_index=True,
                        how='inner')

    def get_dates_min(self):
        """
        Build a DataFrame with extrapolated minimum date per case
        For us the min date is the minimum date on which all selected VOI are available
        :return:
        """
        q = self._queries['QUERY_DT_MIN_ICU'].format(wd.np_array_2_string(self._dic.get_all_voi_codes(self._voi)))
        df_q = self._drill_conn.df_from_query(q)
        df_q = df_q.pivot(index='id_ndaj1', columns='id_measure_type', values='min_dt')
        for v in self._voi:
            if v != 'BT':
                aphp_codes = self._dic.get_all_voi_codes([v])
                sub_df_q = df_q[aphp_codes]
                df_q[v] = pd.to_datetime(sub_df_q.min(axis=1))
        return df_q[self._voi].max(axis=1)

    def get_dates_max(self):
        """
        Build a DataFrame with extrapolated maximum date per case
        For us the max date is the maximum date on which all selected VOI are available
        :return:
        """
        q = self._queries['QUERY_DT_MAX_ICU'].format(wd.np_array_2_string(self._dic.get_all_voi_codes(self._voi)))
        df_q = self._drill_conn.df_from_query(q)
        df_q = df_q.pivot(index='id_ndaj1', columns='id_measure_type', values='max_dt')
        for v in self._voi:
            if v != 'BT':
                aphp_codes = self._dic.get_all_voi_codes([v])
                sub_df_q = df_q[aphp_codes]
                df_q[v] = pd.to_datetime(sub_df_q.max(axis=1))
        return df_q[self._voi].min(axis=1)

    def get_case_stay_interval(self):
        """
        Estimate the stay date interval per case
        :return: counter matrix with added columns ==> dt_min, dt_max and stay interval (dt_max - dt_min)
        """
        df_dt_min = pd.DataFrame(self.get_dates_min())
        df_dt_max = pd.DataFrame(self.get_dates_max())
        df = pd.merge(df_dt_min, df_dt_max, left_index=True, right_index=True, how='inner')
        df.columns = ['dt_min', 'dt_max']
        df['stay_len'] = df['dt_max'] - df['dt_min']
        return df

    def case_filter_by_stay_interval(self, c_matrix, interval_min=6):
        """
        Filter the counter_matrix with the minimum value of stay interval and complete it with additional columns
        dt_min, dt_max, interval_stay
        :param c_matrix: matrix of counters per case and per VOI generated by get_counter_matrix
                            (pandas DataFrame)
        :param interval_min: minimum stay in hours
        :return:
        """
        case_stay_matrix = self.get_case_stay_interval()
        c_matrix = pd.merge(c_matrix, case_stay_matrix, left_index=True, right_index=True, how='inner')
        delta_min = timedelta(hours=interval_min)
        c_matrix = c_matrix[c_matrix['stay_len'] >= delta_min]
        return c_matrix

    def table_exists(self, table_name):
        """
        Check if a table exists or not
        :param table_name: table name
        :return: boolean ==> true if the table exists, else false
        """
        q = self._queries['QUERY_TABLE_EXISTS'].format(table_name)
        result = True
        try:
            self._drill_conn.df_from_query(q)
        except pyodbc.Error:
            result = False
        return result

    def drop_table(self, table_name):
        """
        Drop table
        :param table_name: table name
        :return: None
        """
        q = self._queries['QUERY_DROP_TABLE'].format(table_name)
        try:
            self._drill_conn.conn.execute(q)
        except pyodbc.Error:
            pass

    def create_icu_nda_mouv_ufr_tr(self):
        """
        Create table icu_nda_mouv_ufr_tr if it does not exist
        :return:
        """
        if not self.table_exists("icu_nda_mouv_ufr_tr"):
            try:
                self._drill_conn.conn.execute(self._queries['CREATE_ICU_MOV'])
            except pyodbc.Error:
                pass

    def create_icu_sensor_24(self):
        """
        Create table icu_sensor_24 if it does not exist
        :return:
        """
        if not self.table_exists("icu_sensor_24"):
            try:
                self._drill_conn.conn.execute(self._queries['CREATE_ICU_SENSOR_24'])
            except pyodbc.Error:
                pass

    def create_icu_pat_info(self):
        """
        Create table icu_pat_info if it does not exist
        :return:
        """
        if not self.table_exists("icu_pat_info"):
            try:
                self._drill_conn.conn.execute(self._queries['CREATE_ICU_PAT_INFO'])
            except pyodbc.Error:
                pass

    def create_sensor_util(self):
        """
        Create table icu_sensor_util if it does not exist
        :return:
        """
        if not self.table_exists("icu_sensor_util"):
            try:
                self._drill_conn.conn.execute(self._queries['CREATE_SENSOR_UTIL'])
            except pyodbc.Error:
                pass

    def create_nda_j1_dt_range(self):
        """
        Create table nda_j1_dt_range if it does not exist
        :return:
        """
        if not self.table_exists("nda_j1_dt_range"):
            try:
                self._drill_conn.conn.execute(self._queries['CREATE_DT_RANGE'])
            except pyodbc.Error:
                pass

    def get_dt_ranges(self):
        """
        get the features by case (age, date of death, dates min and max for measures)
        :return: a dataframe
        """
        q = "select * from nda_j1_deces"
        df_ranges = self._drill_conn.df_from_query(q)
        df_ranges['id_nda'] = df_ranges['id_nda'].astype(str)
        df_ranges['j1'] = df_ranges['j1'].astype(str)
        df_ranges['dt_min'] = pd.to_datetime(df_ranges['dt_min'])
        df_ranges['dt_max'] = pd.to_datetime(df_ranges['dt_max'])
        return df_ranges


if __name__ == '__main__':
    dq = DrillQueries("drill_eds", ['HR', 'RR', 'ABPS', 'ABPD', 'SPO2'])

    # print(dq.get_total_cases())
    counter_matrix = dq.get_counter_matrix()
    print("Number of cases for first 24h: {}".format(len(counter_matrix)))
    # print(counter_matrix.head(10))
    counter_matrix = dq.case_first_level_filter(counter_matrix)
    print("Number of cases for first 24h having values for {}: {}".format(dq._voi, len(counter_matrix)))
    counter_matrix = dq.case_second_level_filter(counter_matrix, 6)
    print("Number of cases for first 24h having at least 6 values for {}: {}".format(dq._voi, len(counter_matrix)))
    demographic_df = dq.get_cases_data()
    counter_matrix = dq.case_more_age_filter(counter_matrix, demographic_df, age_min=15)
    # print(counter_matrix.head())
    print("Number of cases for first 24h having at least 6 values for {} and more than 15 \
            years old: {}".format(dq.voi, len(counter_matrix)))
    # mortality rate
    mr = float(counter_matrix[counter_matrix['dt_deces'].notnull()].shape[0]) / float(counter_matrix.shape[0]) * 100.
    print("Mortality rate: {}".format(mr))
    counter_matrix = dq.case_filter_by_stay_interval(counter_matrix, interval_min=3)
    print("Number of cases for first 24h having at least 6 values for {} and more than 15 \
            years old and stayed at least 3 hours: {}".format(dq.voi, len(counter_matrix)))
    mr = float(counter_matrix[counter_matrix['dt_deces'].notnull()].shape[0]) / float(counter_matrix.shape[0]) * 100.
    print("Mortality rate: {}".format(mr))
    print(counter_matrix.head())

    # print(dq.table_exists('icu_sensor_24'))
