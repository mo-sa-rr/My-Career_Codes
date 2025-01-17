function [v,w,x,y,z] = cent_opt(user_loc,user_profile,UAV_loc,sigma2,H,max_speed,min_dist,profile_update_rate)

[~,K] = size(user_loc);
[~,M] = size(UAV_loc);
link_preference = zeros(M,K);


for i = 1:M
    for j = 1:K
       link_preference(i,j) = user_profile(j)*log2(1+(1/(H^2+norm(UAV_loc(:,i)-user_loc(:,j))^2))....
           /(sum(1/(H^2+norm(UAV_loc-user_loc(:,j))^2))-1/(H^2+norm(UAV_loc(:,i)-user_loc(:,j))^2)+sigma2));
    end
end

sched = cent_sched_casadi(link_preference,user_profile);
[~,idx] = max(sched');
sched = zeros(M,K);
for i = 1:M
   sched(i,idx(i)) = 1;
end

[pos,val] = cent_pos(sched,user_loc,user_profile,UAV_loc,sigma2,H,max_speed,min_dist);
if M>1
user_profile = user_profile + profile_update_rate*(~sum(sched)); 
else
user_profile = user_profile + profile_update_rate*(~sched);     
end
v = val;
w = link_preference;
x = sched;
y = pos;
z = user_profile;
end