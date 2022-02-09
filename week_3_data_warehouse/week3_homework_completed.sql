--Question 1
SELECT COUNT(*) 
FROM `ecstatic-splice-339103.trips_data_all.fhv_tripdata`
--Question 2
select count(*) from (Select distinct dispatching_base_num from  `ecstatic-splice-339103.trips_data_all.fhv_tripdata`)
--Question 4
SELECT COUNT(*) 
FROM `ecstatic-splice-339103.trips_data_all.fhv_tripdata`
where dispatching_base_num in ("B00987", "B02060", "B02279") and pickup_datetime>'2019-01-01' and pickup_datetime<'2019-03-31'