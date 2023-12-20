# a base cfg file for testing

alg = dict(
    type="MyTestModule",
    post_hooks=[dict(type="ObjectStateLogger", save_folder="tests/utils/tmp_outputs")],
)
