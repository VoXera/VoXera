import datetime


def log_it(object):
    def wrapper(*args,**kwargs):
        
        dts = datetime.datetime.now()
        print(f"\nAt {dts}, 😊VoXera is here: {object.__name__} - {args} - {kwargs}")
        result = object(*args,**kwargs)
        dts = datetime.datetime.now()
        print(f"At {dts}, 😊VoXera has left from: {object.__name__} - {args} - {kwargs}")

        return result
    return wrapper


