CREATE DATABASE restaurantDB;

CREATE TABLE PredLogs (
    LogDate varchar(255),
    LogTime varchar(255),
    Log varchar(255)
);

CREATE TABLE PredValidLogs (
    LogDate varchar(255),
    LogTime varchar(255),
    Log varchar(255)
);

CREATE TABLE TrainLogs (
    LogDate varchar(255),
    LogTime varchar(255),
    Log varchar(255)
);

CREATE TABLE TrainValidLogs (
    LogDate varchar(255),
    LogTime varchar(255),
    Log varchar(255)
);
select * from PredValidLogs;

DELETE FROM PredLogs;
DELETE FROM PredValidLogs;
DELETE FROM TrainValidLogs;
DELETE FROM TrainLogs;

DROP TABLE PredValidLogs;