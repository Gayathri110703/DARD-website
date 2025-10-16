-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Jul 29, 2025 at 06:00 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `decoy_documents`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `utype` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`, `email`, `utype`, `mobile`) VALUES
('admin', 'admin', 'bgeduscanner@gmail.com', 'admin', 9894442716),
('ta', '1234', '', 'TA', 0);

-- --------------------------------------------------------

--
-- Table structure for table `ins_access`
--

CREATE TABLE `ins_access` (
  `id` int(11) NOT NULL,
  `user` varchar(20) NOT NULL,
  `docid` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `ins_access`
--

INSERT INTO `ins_access` (`id`, `user`, `docid`) VALUES
(1, 'suresh', 1),
(3, 'siva', 2),
(4, 'siva', 9),
(5, 'siva', 10),
(6, 'siva', 3),
(7, 'siva', 4),
(8, 'siva', 5),
(9, 'siva', 6),
(10, 'siva', 7),
(11, 'siva', 8);

-- --------------------------------------------------------

--
-- Table structure for table `ins_data`
--

CREATE TABLE `ins_data` (
  `id` int(11) NOT NULL,
  `document` varchar(30) NOT NULL,
  `status` int(11) NOT NULL,
  `aadhar_no` int(11) NOT NULL,
  `name` int(11) NOT NULL,
  `dob` int(11) NOT NULL,
  `gender` int(11) NOT NULL,
  `address` int(11) NOT NULL,
  `passport_no` int(11) NOT NULL,
  `date_issue` int(11) NOT NULL,
  `date_expiry` int(11) NOT NULL,
  `mobile` int(11) NOT NULL,
  `email` int(11) NOT NULL,
  `photo` int(11) NOT NULL,
  `licence_no` int(11) NOT NULL,
  `nation` int(11) NOT NULL,
  `state` int(11) NOT NULL,
  `birth_place` int(11) NOT NULL,
  `country_code` int(11) NOT NULL,
  `place_issue` int(11) NOT NULL,
  `signature` int(11) NOT NULL,
  `qr_code` int(11) NOT NULL,
  `sdw_of` int(11) NOT NULL,
  `blood_grp` int(11) NOT NULL,
  `pan_no` int(11) NOT NULL,
  `ration_no` int(11) NOT NULL,
  `bar_code` int(11) NOT NULL,
  `voter_id` int(11) NOT NULL,
  `branch` int(11) NOT NULL,
  `card_name` int(11) NOT NULL,
  `chip_code` int(11) NOT NULL,
  `card_no` int(11) NOT NULL,
  `height` int(11) NOT NULL,
  `weight` int(11) NOT NULL,
  `language` int(11) NOT NULL,
  `plan_type` int(11) NOT NULL,
  `policy_type` int(11) NOT NULL,
  `amount` int(11) NOT NULL,
  `death_benefit` int(11) NOT NULL,
  `subsequent` int(11) NOT NULL,
  `billing` int(11) NOT NULL,
  `provience` int(11) NOT NULL,
  `postal_code` int(11) NOT NULL,
  `coverage` int(11) NOT NULL,
  `id_number` int(11) NOT NULL,
  `reg_no` int(11) NOT NULL,
  `make_model` int(11) NOT NULL,
  `year` int(11) NOT NULL,
  `color` int(11) NOT NULL,
  `fax` int(11) NOT NULL,
  `driver_info` int(11) NOT NULL,
  `vehicle_info` int(11) NOT NULL,
  `insurance_info` int(11) NOT NULL,
  `general_info` int(11) NOT NULL,
  `home_info` int(11) NOT NULL,
  `struct_info` int(11) NOT NULL,
  `fund` int(11) NOT NULL,
  `bank` int(11) NOT NULL,
  `family` int(11) NOT NULL,
  `payment` int(11) NOT NULL,
  `biometric` int(11) NOT NULL,
  `fingerprint` int(11) NOT NULL,
  `handwriting` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `ins_data`
--

INSERT INTO `ins_data` (`id`, `document`, `status`, `aadhar_no`, `name`, `dob`, `gender`, `address`, `passport_no`, `date_issue`, `date_expiry`, `mobile`, `email`, `photo`, `licence_no`, `nation`, `state`, `birth_place`, `country_code`, `place_issue`, `signature`, `qr_code`, `sdw_of`, `blood_grp`, `pan_no`, `ration_no`, `bar_code`, `voter_id`, `branch`, `card_name`, `chip_code`, `card_no`, `height`, `weight`, `language`, `plan_type`, `policy_type`, `amount`, `death_benefit`, `subsequent`, `billing`, `provience`, `postal_code`, `coverage`, `id_number`, `reg_no`, `make_model`, `year`, `color`, `fax`, `driver_info`, `vehicle_info`, `insurance_info`, `general_info`, `home_info`, `struct_info`, `fund`, `bank`, `family`, `payment`, `biometric`, `fingerprint`, `handwriting`) VALUES
(1, 'Passport', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(2, 'Aadhar', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(3, 'Driving License', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(4, 'PAN Card', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(5, 'Ration Card', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(6, 'Voter id', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(7, 'Credit Card', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(8, 'Health Insurance', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(9, 'Motor Insurance', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0),
(10, 'Life Insurance', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 0, 0, 0, 0),
(11, 'Home Insurance', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 0),
(12, 'Travel Insurance', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0),
(13, 'Car Insurance', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0);

-- --------------------------------------------------------

--
-- Table structure for table `ins_detect`
--

CREATE TABLE `ins_detect` (
  `id` int(11) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `document` varchar(20) NOT NULL,
  `filename` varchar(50) NOT NULL,
  `date_time` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `ins_detect`
--


-- --------------------------------------------------------

--
-- Table structure for table `ins_files`
--

CREATE TABLE `ins_files` (
  `id` int(11) NOT NULL,
  `user` varchar(20) NOT NULL,
  `document` varchar(30) NOT NULL,
  `filename` varchar(50) NOT NULL,
  `status` int(11) NOT NULL,
  `rdate` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `ins_files`
--

INSERT INTO `ins_files` (`id`, `user`, `document`, `filename`, `status`, `rdate`) VALUES
(1, 'admin', 'docx', 'data1.docx', 0, ''),
(2, 'admin', 'docx', 'data2.docx', 0, ''),
(3, 'admin', 'docx', 'data3.docx', 0, ''),
(4, 'admin', 'docx', 'data4.docx', 0, ''),
(5, 'admin', 'docx', 'data5.docx', 0, ''),
(6, 'admin', 'docx', 'General-Insurance.docx', 0, ''),
(7, 'admin', 'docx', 'ins2.docx', 0, ''),
(8, 'admin', 'docx', 'ins3.docx', 0, ''),
(9, 'admin', 'docx', 'ins4.docx', 0, ''),
(10, 'admin', 'docx', 'sample.docx', 0, '');

-- --------------------------------------------------------

--
-- Table structure for table `ins_register`
--

CREATE TABLE `ins_register` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `gender` varchar(10) NOT NULL,
  `dob` varchar(15) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `city` varchar(20) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL,
  `rdate` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `ins_register`
--

INSERT INTO `ins_register` (`id`, `name`, `gender`, `dob`, `mobile`, `email`, `city`, `uname`, `pass`, `rdate`) VALUES
(1, 'Ravikumar', 'Male', '14-08-1990', 9867845833, 'ravi@gmail.com', 'Chennai', 'ravi', '12345', '26-01-2022');

-- --------------------------------------------------------

--
-- Table structure for table `ins_user`
--

CREATE TABLE `ins_user` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL,
  `rdate` varchar(15) NOT NULL,
  `doc_entry` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `ins_user`
--

INSERT INTO `ins_user` (`id`, `name`, `mobile`, `email`, `uname`, `pass`, `rdate`, `doc_entry`) VALUES
(1, 'Suresh', 9070603011, 'suresh@gmail.com', 'suresh', '12345', '22-02-2025', 0),
(2, 'Gopi', 9070603011, 'gopi@gmail.com', 'gopi', '12345', '22-02-2025', 0),
(3, 'Nisha', 9956744332, 'nisha@gmail.com', 'nisha', '12345', '22-02-2025', 0),
(4, 'Siva', 9054621096, 'siva@gmail.com', 'siva', '1234', '22-03-2025', 1);
