using MIRT: jim
using Images
using  Noise

m = 128; n = 128

#Replicating figure 4 from Double Gaussians
fig4 = ones( (m,n) )*255

x_mid = Int(m/2)
y_fourth = Int(n/4)
y_half = Int(n/2)
y_threefourth = Int((3*n)/4)

val1 = 0
val2 = 80
val3 = 170
val4 = 255

fig4[1:x_mid,1:y_fourth] .= val4
fig4[1:x_mid,y_fourth:y_half] .= val2
fig4[1:x_mid,y_half:y_threefourth] .= val3
fig4[1:x_mid,y_threefourth:end] .= val1

fig4[x_mid:end,1:y_fourth] .= val3
fig4[x_mid:end,y_fourth:y_half] .= val1
fig4[x_mid:end,y_half:y_threefourth] .= val4
fig4[x_mid:end,y_threefourth:end] .= val2

fig4 = fig4'./255

jim(fig4)
save("figure4a.png",colorview(Gray,fig4))

#Add gaussian noise
var = sqrt(0.018)
mean = 0
fig4_gauss = add_gauss(fig4,var, mean)

save("figure4b.png", colorview(Gray,fig4_gauss))

## Figure 7

fig7 = ones((m,n))
