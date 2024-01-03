function r = Generank(W,ex,d)
%function r = geneRank(W,ex,d)
% 
% GeneRank is a modification of the PageRank algorithm.
% input data is   W: connectivity  matrix (zero/one, symmetric with zero diag)
%                 ex: vector of expression levels (non-negative)
%                 d: parameter in algorithm
%
% output is   r: vector of rankings
%
% March 09/2004
%
% Reference: GeneRank: Using search engine technology for the analysis
%            of microarray experiments,       
%            by Julie L. Morrison, Rainer Breitling, 
%            Desmond J. Higham and David R. Gilbert, 
%            submitted for publication.

ex = abs(ex);
norm_ex = ex/max(ex);
w = sparse(W);
degrees = sum(W);
ind = find(degrees== 0);
degrees(ind) = 1;
D1 = sparse(diag(1./degrees));
A = eye(size(w)) - d*(w'*D1);
b = (1-d)*norm_ex;
r = A\b;
