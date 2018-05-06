function [z w v train_err err_v] = mlptrain(train,validate,m,k)
%m = 18 ;
%k = 10;
%train = load('optdigits_train.txt');
%validate = load('optdigits_valid.txt');

%sigmoid = @(x) 1/(1+exp(-x));
relu = @(x) (x>0)*x;

n = length(train);
d = size(train,2) - 1;

w = (rand(m,d+1)-0.5)/50;
v = (rand(k,m+1)-0.5)/50;
z = zeros(n,m);
y = zeros(n,k);
o = zeros(1,k);
eta = 1e-5;

delt_v = zeros(k,m+1);
delt_w = zeros(m,d+1);
r = zeros(1,k);

loop_counter = 30000;
same = 0;
acc = [];
Err = [];
while loop_counter > 0
    for i = 1 : n
        x = [train(i,1:d) 1];
        for j = 1 : m
            z(i,j) = relu(w(j,:) * x');
        end
        Z = [z(i,:) 1];
        for j = 1 : k
            o(j) = v(j,:) * Z';
        end
        y(i,:) = exp(o)/sum(exp(o));
        for j = 1 : k
            r(j) = (j == train(i,d+1)+1);
        end
        for j = 1 : k
            delt_v(j,:) = eta * (r(j) - y(i,j)) * Z;
        end
        for j = 1 : m
            if w(j) * x' < 0
                delt_w(j,:) = 0;
            else
                S = 0;
                for t = 1 : k
                    S = S + (r(t) - y(i,t))*v(t,j);
                end
                delt_w(j,:) = eta * S * x;
            end
        end
        v = v + delt_v;
        w = w + delt_w;
    end
    temp_err = 0;
    for i = 1 : n
        r_idx = train(i,d+1) + 1;
        temp_err = temp_err - log(y(i,r_idx));
    end
    Err = [Err temp_err];
    loop_counter = loop_counter - 1;
    [dump idx] = max(y,[],2);
    same = sum(idx-1 == train(:,d+1));
    acc = [acc same];
    if length(Err)>1&&abs(Err(length(Err))-Err(length(Err)-1))<1e-4
        break
    end
end
train_err = (n-same)/n;
sprintf('m is %d, training error is %.5f',m,train_err)


n = length(validate);
z_v = zeros(1,m);
y_v = zeros(1,k);
err_v = 0;
for i = 1 : n
    x = [validate(i,1:d) 1];
    for j = 1 : m
        z_v(j) = relu(w(j,:) * x');
    end
    Z = [z_v 1];
    for j = 1 : k
        o(j) = v(j,:) * Z';
    end
    y_v = exp(o)/sum(exp(o));
    [dump idx] = max(y_v,[],2);
    err_v = err_v + (idx-1 ~= validate(i,d+1));
    
end
err_v = err_v / n;
sprintf('m is %d,validation error is %.5f',m,err_v)
end
