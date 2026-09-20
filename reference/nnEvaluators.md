# Compiled neural-network evaluators

The \`nn\<K\>()\` family evaluates a network of \`K\` inputs, and
\`nn\<K\>\_d\<j\>()\` / \`nn\<K\>\_d\<j\>\_d\<l\>()\` its first and
second input derivatives. They are generated (see \`tools/genNn.R\`),
registered with rxode2 as model functions, and are not intended to be
called directly – use \[nn()\] in a model and \[nnEval()\] to inspect a
fitted network.

## Arguments

- id:

  integer network id (0-based), the first argument of the model call.

- ...:

  the network inputs.

## Value

numeric vector of the network value or derivative at each input row.
