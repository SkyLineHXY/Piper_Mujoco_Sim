class ClassUtils:
    @staticmethod
    def get_leaf_subclasses(base_class) -> list:
        leaf_subclasses = []
        for subclass in base_class.__subclasses__():
            if subclass.__subclasses__():
                leaf_subclasses.extend(ClassUtils.get_leaf_subclasses(subclass))
            else:
                leaf_subclasses.append(subclass)
        return leaf_subclasses
