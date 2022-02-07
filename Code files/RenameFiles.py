from _pytest import pathlib

for path in pathlib.Path("C:/Users/eliet/OneDrive/Desktop/dd").iterdir():
    if path.is_file():
        old_name = path.stem



        old_extension = path.suffix

        directory = path.parent

        new_name = "re" + old_name + old_extension

        path.rename(pathlib.Path(directory, new_name))



