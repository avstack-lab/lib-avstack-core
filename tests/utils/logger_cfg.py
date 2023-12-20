# a base cfg file for testing

alg = dict(
    type="MyTestModule",
    post_hooks=[dict(type="ObjectLogger", save_folder="tests/utils/tmp_outputs")],
)
