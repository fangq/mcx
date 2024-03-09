## Check List
Before you submit your pull-request, please verify and check all below items

- [ ] You have only modified the minimum number of lines that are necessary for the update; you should never add/remove white spaces to other lines that are irrelevant to the desired change.
- [ ] You have run `make pretty` (requires `astyle` in the command line) under the `src/` folder and formatted your C/C++/CUDA source codes **before every commit**; similarly, you should run `python3 -m black *.py` (`pip install black` first) to reformat all modified Python codes, or run `mh_style --fix .` (`pip install miss-hit` first) at the top-folder to format all MATLAB scripts.
- [ ] Add sufficient in-code comments following the [`doxygen` C format](https://fnch.users.sourceforge.net/doxygen_c.html)
- [ ] In addition to source code changes, you should also update the documentation ([README.md](https://github.com/fangq/mcx/blob/master/README.md), [mcx_utils.c](https://github.com/fangq/mcx/blob/v2023/src/mcx_utils.c#L5029-L5236) and/or [mcxlab.m](https://github.com/fangq/mcx/blob/v2023/mcxlab/mcxlab.m)) if any command line flag was added or changed.

If your commits included in this PR contain changes that did not follow the above guidelines, you are strongly recommended to create a clean patch using `git rebase` and `git cherry-pick` to prevent in-compliant history from appearing in the upstream code.

Moreover, you are highly recommended to 

- Add a test in the `mcx/test/testmcx.sh` script, following existing examples, to test the newly added feature; or add a MATLAB script under `mcxlab/examples` to gives examples of the desired outputs
- MCX's simulation speed is currently limited by the number of GPU registers. In your change, please consider minimizing the use of registers by reusing existing ones. CUDA compilers may not be able to optimize register counts, thus require manual optimization.
- Please create a github Issue first with detailed descriptions of the problem, testing script and proposed fixes, and link the issue with this pull-request

## Please copy/paste the corresponding Issue's URL after the below dash

- 
