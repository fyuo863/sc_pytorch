import inspect

class MyClass:
    def __init__(self):
        print("Object is created.")
    def func1(self):
        print("Function 1 is executed.")
    
    def func2(self):
        print("Function 2 is executed.")
    
    def func3(self):
        print("Function 3 is executed.")
    
    def func4(self):
        print("Function 4 is executed.")
    
    def func5(self):
        print("Function 5 is executed.")
    
    def execute_all(self):
        print("Executing all functions:")
        # 获取类中的所有函数并依次执行
        list = [func for func, obj in inspect.getmembers(MyClass, predicate=inspect.isfunction)]
        print(list)
        list.pop(0)
        list.pop(0)
        print(list)
        for func_name in list:
            func = getattr(self, func_name)
            func()

# 创建对象并执行所有方法
obj = MyClass()
obj.execute_all()
# list = [func for func, obj in inspect.getmembers(MyClass, predicate=inspect.isfunction)]
# list.pop(0)
# print(list)