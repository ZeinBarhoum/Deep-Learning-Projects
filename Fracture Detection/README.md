# Fracture Detection

In this directory, a project for fracture detection is implemented using Tensorflow library and the MURA dataset.

A comparison between Custom CNN models and pre-trained models like VGG, ResNet and Densenet was conducted.

The best model was used in a GUI that takes the X-Ray image and gives the prediction (fracture or not) and the confidence level. The data can be logged into a PostgreSQL database. The database is assumed to be named `fractures` with username `postgres`, the password is required to log the data into the database.

To test the program, run the following command:

```bash
python3 final.py
```

In the `tests` directory, there are some test samples (images) that can be used to test the program.