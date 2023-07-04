===============
Data Processing
===============


Data Cleaning 
-------------

1. Removing no claim data
2. Adding missing weeks with zero sales
3. Merging different data sources to theme level
4. Dividing product sales between theme as per theme count
5. Aggregating data on weekly level
6. Missing values filled with zero for search count
7. Data format made consistent


Data Consolidation
------------------

Merged Product sales data at theme level with maximum corelated latency of search and social data.


Imputation
----------

Filled missing week with zero sale for a theme


Feature transformations
-----------------------

Standard Scaling


Feature engineering
-------------------

1. Divided product sales proportionally to every theme, since a product is available in more than one theme
2. Calculated different lag columns weekly for search and social post and filtered the maximum correlated week for every theme
3. Calculated average price per unit for all the vendors
4. Calculated the number of products for a vendor per theme
