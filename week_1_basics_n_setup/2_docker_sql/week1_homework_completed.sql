--Question 3
select *
from yellow_taxi_data
where EXTRACT(month FROM tpep_pickup_datetime) = 01
--   and EXTRACT(month FROM tpep_dropoff_datetime) = 01
--   and EXTRACT(day FROM tpep_dropoff_datetime) = 15
  and EXTRACT(day FROM tpep_pickup_datetime) = 15

--Question 4
select EXTRACT(day FROM tpep_pickup_datetime) as day_jan, max(tip_amount) as max_amount
from (select *
      from yellow_taxi_data
      where EXTRACT(month FROM tpep_pickup_datetime) = 01
        and EXTRACT(year FROM tpep_pickup_datetime) = 2021) as tt
group by day_jan
order by max_amount

--Question 5
select "DOLocationID", z."Zone", count(passenger_count) as count_id
from (select *
      from yellow_taxi_data
      where EXTRACT(month FROM tpep_pickup_datetime) = 01
        and EXTRACT(year FROM tpep_pickup_datetime) = 2021
        and EXTRACT(day FROM tpep_pickup_datetime) = 14
        and "PULocationID" = 43) as tt
         join zones z on z."LocationID" = tt."DOLocationID"
group by z."Zone", "DOLocationID"
order by count_id desc

--Question 6

select tt.*, z."Zone", z2."Zone" from (
select "PULocationID", "DOLocationID", avg(total_amount) as avg_price from yellow_taxi_data Y
group by "PULocationID", "DOLocationID"
order by avg_price Desc LIMIT 1) as tt
join zones z on z."LocationID"=tt."PULocationID"
join zones z2 on z2."LocationID"=tt."DOLocationID"







