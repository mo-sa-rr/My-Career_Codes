%% sample Online optimization

clear
clc


operation_length = 1000;
sigma2 = 1; % noise power        
H = 10 ; % UAV Height
max_speed = 1.5;
min_dist = 1 ;
K = 8 ; % number of users
M = 3 ; % number of UAVs !!! set M<K
max_loc = 30;
plot_offset = 5;

user_profile = ones(1,K);
profile_update_rate = 0.2;

rng(30); % fixing random generator seed

x = randi([-max_loc,max_loc],1,K);     % initial location of users

rng(21); % fixing random generator seed 
y = randi([-max_loc,max_loc],1,K);

user_location = [x;y];


rng(3);
x = randi([-max_loc,max_loc],1,M); 
rng(5);
y = randi([-max_loc,max_loc],1,M); 

UAV_loc = [x;y];

scatter(user_location(1,:),user_location(2,:),200,'O');
hold on
scatter(UAV_loc(1,:),UAV_loc(2,:),1000,'X','r');
xlim([-max_loc-plot_offset,max_loc+plot_offset]);
ylim([-max_loc-plot_offset,max_loc+plot_offset]);
for j =1:K
    text(user_location(1,j),user_location(2,j),sprintf('%d',j));

end

for i =1:M
    text(UAV_loc(1,i),UAV_loc(2,i),sprintf('%d',i));

end

hold off




str = "========================================================";




cumulative_val = 0;
counter = 0;
while counter<operation_length
    
    pause(0.1);
    tic
    
[V,W,X,Y,Z] = cent_opt(user_location,user_profile,UAV_loc,sigma2,H,max_speed,min_dist,profile_update_rate) ;



user_profile = Z;
if min(user_profile)>5
    user_profile = user_profile-5;
end
    
UAV_loc = Y;

fig = figure;

scatter(user_location(1,:),user_location(2,:),200,'O');
hold on
scatter(UAV_loc(1,:),UAV_loc(2,:),1000,'X');


[~,idx] = max(X');
for j =1:K
    text(user_location(1,j),user_location(2,j),sprintf('%d',j));

end

for i =1:M
    text(UAV_loc(1,i),UAV_loc(2,i),sprintf('%d',i));
    plot([UAV_loc(1,i),user_location(1,idx(i))],[UAV_loc(2,i),user_location(2,idx(i))]);
end

xlim([-max_loc-plot_offset,max_loc+plot_offset]);
ylim([-max_loc-plot_offset,max_loc+plot_offset]);

title("centralized Operation Process," + K + " users," + M + " Agents, & T = " + counter)

 drawnow
 frame = getframe(fig);
 im{counter+1} = frame2im(frame);

hold off
   
str
%link_preference = W
scheduling = X
user_profile
val = V
cumulative_val =cumulative_val + V
counter = counter + 1

toc
end


filename = "main cent.gif"; % Specify the output file name
for idx = 1:counter
    [A,map] = rgb2ind(im{idx},256);
    if idx == 1
       imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',0.1);
    else
        imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',0.1);
    end
end

