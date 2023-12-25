from infer_pipeline_promax import InferModel

model = InferModel(config_path='/home/dapici/editsql/infer_promax_config.json')

print(model.infer(db_id='product_catalog', utterances=['What are the names of catalog with number 8']))