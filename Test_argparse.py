import argparse

parser = argparse.ArgumentParser(description="Type in the file name.")

# # type: is the data type of the parameter to be passed in
# # help: is a prompt message for the parameter
parser.add_argument("file", type=str, help="File name")

args = parser.parse_args()

# Get the incoming parameters
print(args)
