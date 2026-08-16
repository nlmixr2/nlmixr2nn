## Print the directory that holds libtorch's include/ and lib/, or nothing.
##
## Run by ./configure and ./configure.win.  Kept as a file rather than inlined
## into the shell scripts so the two cannot drift apart, and so the quoting is
## readable.
##
## `torch::torch_install_path()` is the authoritative answer -- the R 'torch'
## package can install libtorch somewhere other than its own package directory
## -- but it is only reachable once torch's namespace loads, which fails on a
## partial install.  `system.file()` is the fallback, and needs no namespace.
p <- tryCatch(suppressWarnings(suppressMessages(torch::torch_install_path())),
              error = function(e) "")
if (!is.character(p) || length(p) != 1L || is.na(p) || !nzchar(p)) {
  p <- system.file(package = "torch")
}
if (nzchar(p)) cat(p)
