# Extract the raw datasets into a spark SQL DataFrame
# Read the guidebooks for each of these datasets to understand how to map similar features to each other
# Perform an exact join on these features, and approximate a disease prevalence statistic within the US population

import sys
from pyspark.sql import SparkSession
#feel free to def new functions if you need
from pyspark.sql.functions import when

# to make it easier to see print statements
spark = SparkSession.builder.\
        master('local').\
        appName('foo').\
        getOrCreate()
spark.sparkContext.setLogLevel('WARN')

def create_dataframe(filepath, format, spark):
    """
    Create a spark df given a filepath and format.

    :param filepath: <str>, the filepath
    :param format: <str>, the file format (e.g. "csv" or "json")
    :param spark: <str> the spark session

    :return: the spark df uploaded
    """

    spark_df = spark.read.format(format).load(filepath, header=True)

    return spark_df


def transform_nhis_data(nhis_df):
    """
    Transform df elements (maps the NHIS data onto the BRFSS data so matching NHIS/BRFSS columns have been mapped to BRFSS formatting)

    :param nhis_df: spark df
    :return: spark df, transformed df
    """

    #add your code here
    transformed_df = nhis_df.withColumnRenamed("MRACBPI2","_IMPRACE").withColumnRenamed("AGE_P","_AGEG5YR")

    #format sex (float)
    transformed_df = transformed_df.withColumn("SEX", transformed_df.SEX.cast("float"))

    #format race
    transformed_df = transformed_df.withColumn("_IMPRACE", when(transformed_df._IMPRACE == 3,4.0)
    .when((transformed_df._IMPRACE == 6) | (transformed_df._IMPRACE == 7) | (transformed_df._IMPRACE == 12), 3.0)
    .when((transformed_df._IMPRACE == 16) | (transformed_df._IMPRACE == 17), 6.0)
    .when(transformed_df.HISPAN_I != 12,5.0)
    .otherwise(transformed_df._IMPRACE.cast('float')))
    # transformed_df.show()

    # drop hispanic column since classified as a race in brfss
    transformed_df = transformed_df.drop('HISPAN_I')

    #format age
    transformed_df = transformed_df.withColumn('_AGEG5YR', when((transformed_df._AGEG5YR >= 18) & (transformed_df._AGEG5YR <= 24), 1.0)
    .when((transformed_df._AGEG5YR >= 25) & (transformed_df._AGEG5YR <= 29), 2.0)
    .when((transformed_df._AGEG5YR >= 30) & (transformed_df._AGEG5YR <= 34), 3.0)
    .when((transformed_df._AGEG5YR >= 35) & (transformed_df._AGEG5YR <= 39), 4.0)
    .when((transformed_df._AGEG5YR >= 40) & (transformed_df._AGEG5YR <= 44), 5.0)
    .when((transformed_df._AGEG5YR >= 45) & (transformed_df._AGEG5YR <= 49), 6.0)
    .when((transformed_df._AGEG5YR >= 50) & (transformed_df._AGEG5YR <= 54), 7.0)
    .when((transformed_df._AGEG5YR >= 55) & (transformed_df._AGEG5YR <= 59), 8.0)
    .when((transformed_df._AGEG5YR >= 60) & (transformed_df._AGEG5YR <= 64), 9.0)
    .when((transformed_df._AGEG5YR >= 65) & (transformed_df._AGEG5YR <= 69), 10.0)
    .when((transformed_df._AGEG5YR >= 70) & (transformed_df._AGEG5YR <= 74), 11.0)
    .when((transformed_df._AGEG5YR >= 75) & (transformed_df._AGEG5YR <= 79), 12.0)
    .when(transformed_df._AGEG5YR >= 80, 13.0)
    .when(transformed_df._AGEG5YR.isNull(), 14.0)
    .otherwise(transformed_df._AGEG5YR.cast("float")))
    
    # transformed_df.show()
    return transformed_df


def calculate_statistics(joined_df):
    """
    Calculate prevalence statistics for diabetes by race/ethnic background, gender, & age

    :param joined_df: the joined df

    :return: None
    """

    # race/ethnic background
    white = joined_df.filter(joined_df._IMPRACE == 1.0)
    black = joined_df.filter(joined_df._IMPRACE == 2.0)
    asian = joined_df.filter(joined_df._IMPRACE == 3.0)
    aia = joined_df.filter(joined_df._IMPRACE == 4.0)
    hispanic = joined_df.filter(joined_df._IMPRACE == 5.0)

    # gender
    fem = joined_df.filter(joined_df.SEX == 2.0)
    mal = joined_df.filter(joined_df.SEX == 1.0)

    # age
    one = joined_df.filter((joined_df._AGEG5YR >= 1.0) & (joined_df._AGEG5YR <= 5.0)) # 1-5: 18-44
    two = joined_df.filter((joined_df._AGEG5YR >= 6.0) & (joined_df._AGEG5YR <= 9.0)) # 6-9: 45-64
    three = joined_df.filter((joined_df._AGEG5YR >= 10.0) & (joined_df._AGEG5YR <= 13.0)) # 10-13: 65+

    #DIBEV1: 1=yes, 2=no, 3=borderline/pre, 7/8/9=unclear
    print("      ",(white.filter(white.DIBEV1 == 1).count()/white.count())*100,"% of white people reported having diabetes")
    print("      ",black.filter(black.DIBEV1 == 1).count()/black.count()*100,"% of black people reported having diabetes")
    print("      ",asian.filter(asian.DIBEV1 == 1).count()/asian.count()*100,"% of asian people reported having diabetes")
    print("      ",aia.filter(aia.DIBEV1 == 1).count()/aia.count()*100,"% of native american or alaska native people reported having diabetes")
    print("      ",hispanic.filter(hispanic.DIBEV1 == 1).count()/hispanic.count()*100,"% of hispanic people reported having diabetes")
    print()
    print("      ",fem.filter(fem.DIBEV1 == 1).count()/fem.count()*100,"% of females reported having diabetes")
    print("      ",mal.filter(mal.DIBEV1 == 1).count()/mal.count()*100,"% of males reported having diabetes")
    print()
    print("      ",one.filter(one.DIBEV1 == 1).count()/one.count()*100,"% of people ages 18-44 reported having diabetes")
    print("      ",two.filter(two.DIBEV1 == 1).count()/two.count()*100,"% of people ages 45-64 reported having diabetes")
    print("      ",three.filter(three.DIBEV1 == 1).count()/three.count()*100,"% of people ages 65 and older reported having diabetes")


def join_data(brfss_df, nhis_df):
    """
    Join dataframes

    :param brfss_df: spark df
    :param nhis_df: spark df after transformation
    :return: the joined df

    """
    #add your code here
    joined_df = brfss_df.join(nhis_df, on=['SEX', '_IMPRACE', '_AGEG5YR'], how="inner")
    # assume each combination is unique while joining, i.e. no duplication
    
    # brfss_df.show()
    # nhis_df.show()
    # joined_df.show()

    # drop any rows containing null values
    joined_df = joined_df.dropna() # not necessarily correct to do in practice

    #print("      SHAPE:", joined_df.count())

    return joined_df

if __name__ == '__main__':

    brfss_filename = sys.argv[1]
    nhis_filename = sys.argv[2]
    save_output = sys.argv[3]
    if save_output == "True":
        output_filename = sys.argv[4]
    else:
        output_filename = None
    

    # Start spark session
    spark = SparkSession.builder.getOrCreate()

    # Load dataframes
    brfss_df = create_dataframe(brfss_filename, 'json', spark)
    nhis_df = create_dataframe(nhis_filename, 'csv', spark)
    # Perform mapping on nhis dataframe
    nhis_df = transform_nhis_data(nhis_df)
    # Join brfss and nhis df
    joined_df = join_data(brfss_df, nhis_df)
    # Save
    if save_output == "True":
        joined_df.write.csv(output_filename, mode='overwrite', header=True)
    # Calculate and print statistics
    calculate_statistics(joined_df)
    
    
    # Stop spark session
    spark.stop()

# spark-submit p1.py test_brfss.json test_nhis.csv True output
# spark-submit p1.py brfss_input.json nhis_input.csv True output
# spark-submit p1.py brfss_input.json nhis_input.csv False