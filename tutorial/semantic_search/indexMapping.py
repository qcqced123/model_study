""" Dict Object for Index Mapping in elasticsearch """
indexName = "all_products"

indexMapping = {
    "properties":{
        "ProductID":{
            "type":"long"
        },
        "ProductName":{
            "type":"text"
        },
        "ProductBrand":{
            "type":"text"
        },
        "Gender":{
            "type":"text"
        },
        "Price (INR)":{
            "type":"long"
        },
        "NumImages":{
            "type":"long"
        },
        "Description":{
            "type":"text"
        },
        "PrimaryColor":{
            "type":"text"
        },
        "DescriptionVector":{
            "type":"dense_vector",
            "dims": 384,
            "index":True,
            "similarity": "cosine"
        }

    }
}