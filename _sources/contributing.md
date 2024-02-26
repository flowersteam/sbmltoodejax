# Contributing 

## All contributions are welcome!
SBMLtoODEjax is in its early stage and any sort of contribution will be highly appreciated.
There are many ways to contribute, including:

- Raising [issues](https://github.com/flowersteam/sbmltoodejax/issues) related to bugs
  or desired enhancements.
- Contributing or improving the
  [docs](https://github.com/flowersteam/sbmltoodejax/tree/main/docs/source/) or
  [examples](https://github.com/flowersteam/sbmltoodejax/tree/main/docs/source/tutorials).
- Fixing [issues](https://github.com/flowersteam/sbmltoodejax/issues).
- Extending or improving our [codebase](https://github.com/flowersteam/sbmltoodejax/tree/main/sbmltoodejax) or [unit tests](https://github.com/flowersteam/sbmltoodejax/tree/main/test).



## How can I contribute to the source code?

Submitting code contributions to SBMLtoODEjax is done via a [GitHub pull
request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).
Our preferred workflow is to first fork the [GitHub
repository](https://github.com/flowersteam/sbmltoodejax/), clone it to your local
machine, and develop on a _feature branch_. Once you're happy with your changes,
`commit` and `push` your code.

**New to this?** Don't panic, our step-by-step-guide below will walk
you through every detail!

### Step-by-step guide

1.  Click [here](https://github.com/flowersteam/sbmltoodejax/fork) to Fork SBMLtoODEjax's
  codebase (alternatively, click the 'Fork' button towards the top right of
  the [main repository page](https://github.com/flowersteam/sbmltoodejax/)). This
  adds a copy of the codebase to your GitHub user account.

2.  Clone your SBMLtoODEjax fork from your GitHub account to your local disk, and add
  the base repository as a remote:
  ```bash
  $ git clone git@github.com:<your GitHub handle>/sbmltoodejax.git
  $ cd sbmltoodejax
  $ git remote add upstream git@github.com:<your GitHub handle>/sbmltoodejax.git
  ```

3. Install the sbmltoodejax package. We suggest using a virtual environment for development. 
Once the virtual environment is activated, run:

  ```bash
  $ pip install -e .
  ```

4. Create a `feature` branch to hold your development changes:

  ```bash
  $ git checkout -b my-feature
  ```
  Always use a `feature` branch. It's good practice to avoid
  work on the ``main`` branch of any repository.

5.  Add changed files using `git add` and then `git commit` files to record your
  changes locally:

  ```bash
  $ git add modified_files
  $ git commit
  ```
  After committing, it is a good idea to sync with the base repository in case
  there have been any changes:

  ```bash
  $ git fetch upstream
  $ git rebase upstream/main
  ```

  Then push the changes to your GitHub account with:

  ```bash
  $ git push -u origin my-feature
  ```

7.  Go to the GitHub web page of your fork of the SBMLtoODEjax repo. Click the 'Pull
  request' button to send your changes to the project's maintainers for
  review.

:::{note}
This guide was derived from [GPJax's guide to
contributing](https://docs.jaxgaussianprocesses.com/contributing/).
:::