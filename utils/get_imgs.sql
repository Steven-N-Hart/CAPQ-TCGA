WITH specimen_preparation_sequence_items AS (
  SELECT DISTINCT
    SeriesInstanceUID AS digital_slide_id,
    steps_unnested2.ConceptNameCodeSequence[SAFE_OFFSET(0)].CodeMeaning AS item_name,
    steps_unnested2.ConceptCodeSequence[SAFE_OFFSET(0)].CodeMeaning AS item_value
  FROM
    `bigquery-public-data.idc_v11.dicom_all`
    CROSS JOIN
      UNNEST(SpecimenDescriptionSequence[SAFE_OFFSET(0)].SpecimenPreparationSequence) AS steps_unnested1
    CROSS JOIN
      UNNEST(steps_unnested1.SpecimenPreparationStepContentItemSequence) AS steps_unnested2
),

grouped_by_study AS (
  SELECT
    dicom.StudyInstanceUID AS case_id,
    ANY_VALUE(dicom.ContainerIdentifier) AS physical_slide_id
  FROM
    `bigquery-public-data.idc_v18.dicom_all` AS dicom
  WHERE
    dicom.Modality = 'SM' AND
    EXISTS (
      SELECT 1
      FROM specimen_preparation_sequence_items AS specimen
      WHERE dicom.SeriesInstanceUID = specimen.digital_slide_id
        AND specimen.item_value LIKE '%eosin%'
    )
  GROUP BY
    dicom.StudyInstanceUID
),

ranked_slides AS (
  SELECT
    CAST(dicom.NumberOfFrames AS INT64) AS NumFrames,
    dicom.SeriesInstanceUID AS SeriesInstanceUID,
    dicom.StudyInstanceUID AS StudyInstanceUID,
    dicom.SOPInstanceUID AS SOPInstanceUID,
    dicom.ContainerIdentifier AS ContainerIdentifier,
    dicom.PatientID AS PatientID,
    dicom.TotalPixelMatrixColumns AS TotalPixelMatrixColumns,
    dicom.TotalPixelMatrixRows AS TotalPixelMatrixRows,
    dicom.collection_id,
    dicom.crdc_instance_uuid,
    dicom.gcs_url,
    dicom.gcs_bucket,
    CAST(dicom.SharedFunctionalGroupsSequence[SAFE_OFFSET(0)].PixelMeasuresSequence[SAFE_OFFSET(0)].PixelSpacing[SAFE_OFFSET(0)] AS FLOAT64) AS pixel_spacing,
    CASE dicom.TransferSyntaxUID
      WHEN '1.2.840.10008.1.2.4.50' THEN 'jpeg'
      WHEN '1.2.840.10008.1.2.4.91' THEN 'jpeg2000'
      ELSE 'other'
    END AS compression,
    specimen.item_name,
    specimen.item_value,
    ROW_NUMBER() OVER (PARTITION BY dicom.StudyInstanceUID ORDER BY CAST(dicom.NumberOfFrames AS INT64) ASC) AS row_num_asc,
    ROW_NUMBER() OVER (PARTITION BY dicom.StudyInstanceUID ORDER BY CAST(dicom.NumberOfFrames AS INT64) DESC) AS row_num_desc
  FROM
    grouped_by_study AS study
  JOIN
    `bigquery-public-data.idc_v18.dicom_all` AS dicom
  ON
    dicom.StudyInstanceUID = study.case_id
    AND dicom.ContainerIdentifier = study.physical_slide_id
  LEFT JOIN
    specimen_preparation_sequence_items AS specimen
  ON
    dicom.SeriesInstanceUID = specimen.digital_slide_id
  WHERE
    dicom.Modality = 'SM' AND
    dicom.collection_name = 'TCGA-BRCA' AND
    dicom.SharedFunctionalGroupsSequence[SAFE_OFFSET(0)].WholeSlideMicroscopyImageFrameTypeSequence[SAFE_OFFSET(0)].FrameType[SAFE_OFFSET(2)] = 'VOLUME'
)

SELECT
  NumFrames,
  SeriesInstanceUID,
  StudyInstanceUID,
  ContainerIdentifier,
  PatientID,
  TotalPixelMatrixColumns,
  TotalPixelMatrixRows,
  collection_id,
  crdc_instance_uuid,
  gcs_url,
  gcs_bucket,
  pixel_spacing,
  compression,
  item_name,
  item_value,
  row_num_asc,
  row_num_desc
FROM
  ranked_slides
WHERE
  (row_num_asc = 1 OR row_num_desc = 1)
ORDER BY
  NumFrames DESC