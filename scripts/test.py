import sys
print("Python:", sys.version)
try:
    import pandas, numpy, simpy, sklearn, tensorflow as tf
    print("pandas:", pandas.__version__)
    print("numpy:", numpy.__version__)
    print("simpy:", simpy.__version__)
    import sklearn as sk
    print("scikit-learn:", sk.__version__)
    print("tensorflow:", tf.__version__)
    print("OK: All core packages imported successfully.")
except Exception as e:
    print("ERROR importing packages:", e)
    raise
