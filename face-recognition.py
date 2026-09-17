import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from atlas_local import LocalDeployment
    import pymongo
    from pymongo.operations import SearchIndexModel
    from pymongo.errors import CollectionInvalid
    import os
    import json
    import insightface
    import cv2
    import time
    import numpy as np

    return (
        CollectionInvalid,
        LocalDeployment,
        SearchIndexModel,
        cv2,
        insightface,
        json,
        mo,
        np,
        os,
        pymongo,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Set up atlas
    Set up an atlas local environment called `image-recognition` using the default settings.

    `LocalDeployment.get_or_create` is idempotent, so we can safely re-run this cell over and over again.
    """)
    return


@app.cell
def _(LocalDeployment, pymongo):
    image_recognition_deployment = LocalDeployment.get_or_create(name="image-recognition")
    image_recognition_deployment.start()
    connection_string = image_recognition_deployment.connection_string()
    client = pymongo.MongoClient(connection_string)
    return (client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Set up the database
    Create the `face_recognition` database and `people` collection.

    MongoDB creates the database and collection lazily, but since we want to create indexes on it we have to make sure it exists.

    We'll create the following indexes:
    - Unique index, ensuring the `first_name` and `last_name` combo is unique
    - Vector index, the start of our show, which will allow us to query the face embeddings (vector)
    """)
    return


@app.cell
def _(CollectionInvalid, SearchIndexModel, client, time):
    db = client.face_recognition

    # 1. collection must exist before a search index can be created on it
    try:
        db.create_collection("people")
    except CollectionInvalid:
        pass

    people = db["people"]

    # 2. unique index on the name pair
    people.create_index(
        [("first_name", 1), ("last_name", 1)], name="name_idx", unique=True
    )

    # 3. vector index
    if "face_vector_index" not in {i["name"] for i in people.list_search_indexes()}:
        people.create_search_index(
            model=SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "face_embedding",
                            "numDimensions": 512,
                            "similarity": "dotProduct",
                        }
                    ]
                },
                name="face_vector_index",
                type="vectorSearch",
            )
        )

    # 4. wait until queryable
    while True:
        info = next(iter(people.list_search_indexes("face_vector_index")), None)
        if info and info.get("queryable"):
            break
        print(".", end="")
        time.sleep(3)

    print("vector index ready")
    return (people,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load dataset

    We're loading the data from the dataset, every person is:
    - `[IMG_NN].json` ➡ Metadata: first name, last name
    - `[IMG_NN].jpg` ➡ Picture of the person

    Steps:
    1. Check if the person is in the database (by `first_name` and `last_name`)
        - if yes: skip
    2. Calculate face embeddings
    3. Insert the person meta + face embeddings in the database
    """)
    return


@app.cell
def _(cv2, insightface, json, os, people):
    app = insightface.app.FaceAnalysis('buffalo_l')
    app.prepare(ctx_id=0)

    # Loop through the dataset directory
    for entry in os.scandir("./_dataset"):
        # Search for the metadata files
        if entry.is_file() and entry.path.endswith(".json"):
            # Open and read the file
            with open(entry.path, "r") as meta_file:
                meta = json.load(meta_file)
                first_name = meta["first_name"]
                last_name = meta["last_name"]

                # Try to find the person in the db, if not found calculate the embeddings and insert in the DB
                if people.find_one(
                    {"first_name": first_name, "last_name": last_name},
                    projection={"_id": 1},
                ) is None:
                    print(f"preparing {first_name} {last_name} embeddings")
                    img = cv2.imread(entry.path.replace(".json", ".jpg"))
                    faces = app.get(img)

                    people.insert_one({
                        "first_name": first_name,
                        "last_name": last_name,
                        "face_embedding": [float(x) for x in faces[0].normed_embedding],
                    })
            
                    print(f"inserted {first_name} {last_name} embeddings in db")
                else:
                    print(f"skipping {first_name} {last_name}, already in db")
    return (app,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Use the vector index

    Steps:
    1. Select a picture
    2. Calculate the embeddings
    3. Use vector search to find the 3 closest people
    """)
    return


@app.cell
def _(mo):
    selected_file = mo.ui.file(filetypes=[".jpg", ".jpeg", ".png"], multiple=False);
    selected_file
    return (selected_file,)


@app.cell
def _(app, cv2, mo, np, people, selected_file):
    # ensure a file is selected
    mo.stop(not selected_file.value, mo.md("Upload an image to continue."))

    # convert the image bytes to a numpy array so it can be read by opencv
    file_bytes = np.frombuffer(selected_file.value[0].contents, np.uint8)
    lookup_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # detect the faces using insightface
    lookup_faces = app.get(lookup_img)

    # helper to find the closest match in the db
    # returns None if no match is found
    def lookup_face(face) -> str | None:
        COS_THRESHOLD = 0.30
        SCORE_THRESHOLD = (COS_THRESHOLD + 1) / 2

        # vector search for the desired face (exact returns the best match)
        # only return the face if the score is > SCORE_THRESHOLD
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "face_vector_index",
                    "path": "face_embedding",
                    "queryVector": face.normed_embedding.tolist(),
                    "limit": 1,
                    "exact": True,
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"score": {"$gte": SCORE_THRESHOLD}}},
            {
                "$project": {
                    "_id": 0,
                    "first_name": 1,
                    "last_name": 1,
                    "cos": {"$subtract": [{"$multiply": ["$score", 2]}, 1]},
                }
            },
        ]

        # Execute the aggregate and get the first result
        result = next(iter(people.aggregate(pipeline)), None)
        if result is None:
            return None

        return f"{result['first_name']} {result['last_name']}"

    # create a new output image
    annotated = lookup_img.copy()

    # look up all the faces found in the image
    for face in lookup_faces:
        # look up the face in the db
        found_face = lookup_face(face)
        is_found = found_face is not None
        rectangle_color = (0, 255, 0) if is_found else (0, 0, 255)

        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), rectangle_color, 2)

        # if the face is found, draw the name
        if is_found:
            # keep the label on-screen if the box is near the top edge
            text_y = y1 - 10 if y1 - 10 > 20 else y2 + 25
            cv2.putText(
                annotated, found_face, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )
 
    mo.image(cv2.imencode(".png", annotated)[1].tobytes())
    return


if __name__ == "__main__":
    app.run()
