.. _overview:

Code overview
=============


**get_nebular_emission** is a Python code to calculate the luminosities of spectral emission lines from global galactic properties, from either a galaxy or a galaxy component.

Purpose and Capabilities
------------------------

This code calculates emission line luminosities from both:

* Star forming HII regions
* Narrow-line regions (NLR) of AGNs

The calculation for star-forming regions follows Baugh et al. 2022, requiring:

* Stellar mass
* Star formation rate (SFR)  
* Cold gas metallicity

The calculation for NLR of AGNs requires:

* Cold gas metallicity in the center
* Bolometric luminosity (or properties to derive this)

Workflow
--------

.. include:: ../README.rst
   :start-after: |flowchart|
   :end-before: Requirements and Installation

Input/Output
------------

The code expects text or HDF5 files with global galactic properties data. It outputs HDF5 files with all calculated results, making it compatible with various galaxy modeling approaches including hydrodynamical simulations and semi-analytical models.

