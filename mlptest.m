function [z err_v] = mlptrain(test,w,v)

%test = load('optdigits_test.txt');
n = length(test);
same = 0;

m = size(w,1);
k = size(v,1);
d = size(w,2) - 1;
relu = @(x) (x>0)*x;

z = zeros(n,m);

z_v = zeros(1,m);
y_v = zeros(1,k);
err_v = 0;
for i = 1 : n
    x = [test(i,1:d) 1];
    for j = 1 : m
        z_v(j) = relu(w(j,:) * x');
    end
    z(i,:) = z_v;
    Z = [z_v 1];
    for j = 1 : k
        o(j) = v(j,:) * Z';
    end
    y_v = exp(o)/sum(exp(o));
    [dump idx] = max(y_v,[],2);
    err_v = err_v + (idx-1 ~= test(i,d+1));
end
err_v = err_v / n;
sprintf('m is %d, test error rate = %f ',m,err_v)

end