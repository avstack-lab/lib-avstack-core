# Contributing

Following the [kitto][kitto] guidelines.

If you discover issues, have ideas for improvements or new features,
please report them to the [issue tracker][issue-tracker] of the repository or
submit a pull request. Please, try to follow these guidelines when you
do so.

## Issue reporting

* Check that the issue has not already been reported.
* Check that the issue has not already been fixed in the latest code
  (a.k.a. `main`).
* Be clear, concise and precise in your description of the problem.
* Open an issue with a descriptive title and a summary using grammatically correct,
  complete sentences. Follow the issue templates.
* Mention the version of the package you are using.
* Include any relevant code to the issue summary.

## Pull requests

* Read [how to properly contribute to open source projects on Github][fork-how].
* Fork the project.
* Use a topic/feature branch to easily amend a pull request later, if necessary.
* Comply with our [git style guide][git-style-guide].
* Use the same coding conventions as the rest of the project.
* Commit and push until you are happy with your contribution.
* Make sure to add tests for it. This is important so I don't break it
  in a future version unintentionally.
* Add an entry to the [Changelog](CHANGELOG.md) accordingly (read: [packaging guidelines][packaging-guidelines]).
* Make sure the test suite is passing
* Try not to decrement the test coverage, unless absolutely necessary.
* [Squash related commits together][squash-rebase] and rebase on upstream main.
* Open a [pull request][using-pull-requests] that relates to *only* one subject 
  with a clear title and description in grammatically correct, complete sentences.


[kitto]: https://github.com/kittiframework/kitto
[issue-tracker]: https://github.com/avstack-lab/lib-avstack-api/issues
[fork-how]: http://gun.io/blog/how-to-github-fork-branch-and-pull-request
[git-style-guide]: https://github.com/agis-/git-style-guide
[using-pull-requests]: https://help.github.com/articles/using-pull-requests
[squash-rebase]: http://gitready.com/advanced/2009/02/10/squashing-commits-with-rebase.html
[issue-template]: https://github.com/kittoframework/kitto/blob/master/ISSUE_TEMPLATE.md
[packaging-guidelines]: https://zorbash.com/post/software-packaging-guidelines
