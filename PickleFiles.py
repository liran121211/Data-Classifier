from datetime import datetime
import pickle


def storeData(obj_model):
    """
    Store object as (pk) dump file
    :param obj_model: object model
    :return: pickled file
    """
    time_stamp = datetime.today().strftime('%d_%m_%Y')
    file_name = time_stamp + '_' + obj_model.__class__.__name__ + '.pk'
    # create empty object file
    model_file = open(file="myFiles\\" + file_name, mode='ab')

    # load (obj_model) data into (model_file)
    pickle.dump(obj=obj_model, file=model_file)
    model_file.close()
    print("[{0}] file saved successfully!\n".format(file_name))


def loadData(file_model):
    """
    Load object from (pk) dump file
    :param file_model: object file
    :return: reconstructed object file
    """
    object_file = None
    try:
        object_file = open(file_model, 'rb')
    except FileNotFoundError:
        print("FATAL ERROR: file does not exist in the given destination.")
    object_model = pickle.load(object_file)
    object_file.close()
    return object_model
