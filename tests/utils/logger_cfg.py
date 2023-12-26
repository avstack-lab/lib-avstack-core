# a base cfg file for testing

alg = dict(
    type="MyTestModule",
    post_hooks=[
        dict(type="ObjectStateLogger", output_folder="tests/utils/tmp_outputs")
    ],
)
