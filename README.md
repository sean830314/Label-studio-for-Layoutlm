# Label Studio For LayoutLM

## Script
load data to label-studio project and load ImageMetaData, ImageData to mongodb
```
LS_QA_ENDPOINT=<LABEL_STUDIO_ENDPOINT> LS_QA_TOKEN=<LABEL_STUDIO_TOKEN> LS_QA_PII_PROJECT_ID=<LABEL_STUDIO_PROJECT> MONGODB_USERNAME=<MONGODB_USERNAME> MONGODB_PASSWORD=<MONGODB_PASSWORD> python ls_loader_ocr_data.py --mongo_host <MONGODB_HOST>   -d 2022-12-16 -f images
```
export combine datasets of label-studio annotated data and ImageData
```
MONGODB_USERNAME=<MONGODB_USERNAME> MONGODB_PASSWORD=<MONGODB_PASSWORD> LS_QA_ENDPOINT=<LABEL_STUDIO_ENDPOINT> LS_QA_TOKEN=<LABEL_STUDIO_TOKEN> LS_QA_PII_PROJECT_ID=<LABEL_STUDIO_PROJECT>  python3 ls_exporter_combine_data.py  --mongo_host <MONGODB_HOST>
```


## Flowchart

```mermaid
graph TD
    A[Prepare data] --> B(Enable Label-studio project)
    B --> |trigger|C(layoutlmv3_data_loader.py)
    C --> |write ImageMetaData and ImageData to databases| D(MongoDB)
    C --> |import new task_id record to label-studio project| E{labeling data on Label Studio}
    C --> |check task_id|C
    E --> |no|E
    E --> |yes, trigger|F(layoutlmv3_combine.py)
    F --> |fetch ImageMetadata and ImageData|D
    F --> |generate data|G(Done)
```

## Schema

```mermaid
classDiagram
  class ImageMetaData {
    -filename : string
    -task_id: string uuid
    -project_id: integer id
    -type: train or test
    -text: document fulltext
  }
  class ImageData {
    -task_id: string uuid
    -project_id: integer id
    -token: string of list
    -bbox: 4-tuple of list
  }
  class AnnotatedImageData {
    -task_id: string uuid
    -label_studio_export_data: LabelStudioExportAnnotations
  }
  ImageMetaData "1" -- "1" ImageData : link by task_id
  ImageMetaData "1" -- "1" AnnotatedImageData : link by task_id
```
