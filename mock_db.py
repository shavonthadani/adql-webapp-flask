import sqlite3
import threading
import queue

def db_worker(db_queue, db_path):
    # Connect to the SQLite database in the worker thread
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    while True:
        query, args, result_queue = db_queue.get()
        if query is None:
            break
        try:
            c.execute(query, args)
            if query.strip().upper().startswith("SELECT"):
                result = c.fetchall()
            else:
                conn.commit()
                result = c.rowcount
            result_queue.put((True, result))
        except Exception as e:
            result_queue.put((False, str(e)))
    conn.close()

def create_tables(db_queue):
    result_queue = queue.Queue()
    db_queue.put(('''
        CREATE TABLE IF NOT EXISTS observation (
          observationURI CHAR ,
          sequenceNumber INT,
          metaReadGroups CHAR,
          proposal_keywords CHAR,
          target_standard INT,
          target_redshift DOUBLE,
          target_moving INT,
          target_keywords CHAR,
          targetPosition_equinox DOUBLE,
          targetPosition_coordinates_cval1 DOUBLE,
          targetPosition_coordinates_cval2 DOUBLE,
          telescope_geoLocationX DOUBLE,
          telescope_geoLocationY DOUBLE,
          telescope_geoLocationZ DOUBLE,
          telescope_keywords CHAR,
          instrument_keywords CHAR,
          environment_seeing DOUBLE,
          environment_humidity DOUBLE,
          environment_elevation DOUBLE,
          environment_tau DOUBLE,
          environment_wavelengthTau DOUBLE,
          environment_ambientTemp DOUBLE,
          environment_photometric INT,
          members CHAR,
          typeCode CHAR,
          metaProducer CHAR,
          metaChecksum CHAR,
          accMetaChecksum CHAR,
          obsID CHAR PRIMARY KEY,
          collection CHAR,
          observationID CHAR,
          algorithm_name CHAR,
          type CHAR,
          intent CHAR,
          metaRelease CHAR,
          proposal_id CHAR,
          proposal_pi CHAR,
          proposal_project CHAR,
          proposal_title CHAR,
          target_name CHAR,
          target_targetID CHAR,
          target_type CHAR,
          targetPosition_coordsys CHAR,
          telescope_name CHAR,
          requirements_flag CHAR,
          instrument_name CHAR,
          lastModified CHAR,
          maxLastModified CHAR
      )

    ''', (), result_queue))

    db_queue.put(('''
        CREATE TABLE IF NOT EXISTS plane (
          publisherID CHAR,
          planeURI CHAR PRIMARY KEY,
          creatorID CHAR,
          obsID CHAR REFERENCES observation(obsID),
          metaReadGroups CHAR,
          dataReadGroups CHAR,
          calibrationLevel INT,
          provenance_keywords CHAR,
          provenance_inputs CHAR,
          metrics_sourceNumberDensity DOUBLE,
          metrics_background DOUBLE,
          metrics_backgroundStddev DOUBLE,
          metrics_fluxDensityLimit DOUBLE,
          metrics_magLimit DOUBLE,
          position_bounds CHAR,
          position_bounds_samples DOUBLE,
          position_bounds_size DOUBLE,
          position_resolution DOUBLE,
          position_sampleSize DOUBLE,
          position_dimension_naxis1 LONG,
          position_dimension_naxis2 LONG,
          position_timeDependent INT,
          energy_bounds_samples DOUBLE,
          energy_bounds_lower DOUBLE,
          energy_bounds_upper DOUBLE,
          energy_bounds_width DOUBLE,
          energy_dimension LONG,
          energy_resolvingPower DOUBLE,
          energy_sampleSize DOUBLE,
          energy_freqWidth DOUBLE,
          energy_freqSampleSize DOUBLE,
          energy_restwav DOUBLE,
          time_bounds_samples DOUBLE,
          time_bounds_lower DOUBLE,
          time_bounds_upper DOUBLE,
          time_bounds_width DOUBLE,
          time_dimension LONG,
          time_resolution DOUBLE,
          time_sampleSize DOUBLE,
          time_exposure DOUBLE,
          polarization_dimension LONG,
          custom_bounds_samples DOUBLE,
          custom_bounds_lower DOUBLE,
          custom_bounds_upper DOUBLE,
          custom_bounds_width DOUBLE,
          custom_dimension LONG,
          metaProducer CHAR,
          metaChecksum CHAR,
          accMetaChecksum CHAR,
          planeID CHAR,
          productID CHAR,
          metaRelease CHAR,
          dataRelease CHAR,
          dataProductType CHAR,
          provenance_name CHAR,
          provenance_version CHAR,
          provenance_reference CHAR,
          provenance_producer CHAR,
          provenance_project CHAR,
          provenance_runID CHAR,
          provenance_lastExecuted CHAR,
          observable_ucd CHAR,
          quality_flag CHAR,
          position_resolutionBounds DOUBLE,
          energy_bounds DOUBLE,
          energy_resolvingPowerBounds DOUBLE,
          energy_emBand CHAR,
          energy_energyBands CHAR,
          energy_bandpassName CHAR,
          energy_transition_species CHAR,
          energy_transition_transition CHAR,
          time_bounds DOUBLE,
          time_resolutionBounds DOUBLE,
          polarization_states CHAR,
          custom_ctype CHAR,
          custom_bounds DOUBLE,
          lastModified CHAR,
          maxLastModified CHAR
      )
    ''', (), result_queue))

    return result_queue.get(), result_queue.get()

def run_query(db_queue, query, args=()):
    result_queue = queue.Queue()
    db_queue.put((query, args, result_queue))
    success, result = result_queue.get()
    if not success:
        raise Exception(result)
    return result