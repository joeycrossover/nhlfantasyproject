# DATA 670 - Capstone, UMGC 2022
# Written by Joseph Coleman
# Last updated March 11, 2022

# The purpose of this script is to perform data pre-processing and exploratory analysis 
# on daily fantasy hockey statistics from the 2021-2022 NHL season, through 3/10/2022.

# Directory Set-up. --------------------------------------------------------
setwd("C:\\Users\\jcole\\Documents\\DATA670\\dailyfantasynerd_nhl_exports")
dir()

# Loading 'goalies' Data Set. -------------------------------------------------------
goalies <- read.csv(file="dfn-goalies-2021-2022.csv", head=TRUE, sep=",", as.is=FALSE)

# 'goalies' data set pre-processing and cleaning. The data was reduced from 1,783 observations to 1,690. --------------------------
apply(goalies, 2, function(goalies) sum(is.na(goalies)))
goalies$Likes[is.na(goalies$Likes)] <- 0
goalies$Likes <- as.integer(goalies$Likes)
goalies <- na.omit(goalies)
goalies <- subset(goalies, Rest != "n/a")
goalies$Rest <- as.integer(as.character(goalies$Rest))
goalies$Vegas.Odds.Win <- gsub("%", "", as.character(goalies$Vegas.Odds.Win))
goalies$Vegas.Odds.Win <- as.numeric(goalies$Vegas.Odds.Win)
goalies$Salary <- as.numeric(goalies$Salary)
goalies$Opp <- gsub("@","", as.character(goalies$Opp))
goalies$Player.Name <- as.character(goalies$Player.Name)
goalies$Team <- as.character(goalies$Team)

# Remove Rows where 'Proj.FP' = 0. The number of observations is now 1,590. -----------------------------------------
goalies <- subset(goalies, Proj.FP != 0)
goalies <- subset(goalies, Actual.FP != 0)

# Loading 'skaters' Data Set. -------------------------------------------------------
skaters <- read.csv(file="dfn-skaters-2021-2022.csv", head=TRUE, sep=",", as.is=FALSE)

# 'skaters' data set pre-processing and cleaning. The data was reduced from 78,806 observations to 55,504. --------------------------
apply(skaters, 2, function(skaters) sum(is.na(skaters)))
skaters$Likes[is.na(skaters$Likes)] <- 0
skaters$Likes <- as.integer(skaters$Likes)
apply(skaters, 2, function(skaters) sum(is.na(skaters)))
skaters <- na.omit(skaters)
skaters <- subset(skaters, Rest != "n/a")
skaters$Rest <- as.integer(as.character(skaters$Rest))
skaters$Salary <- as.numeric(skaters$Salary)
skaters$Opp <- gsub("@","", as.character(skaters$Opp))
skaters$Player.Name <- as.character(skaters$Player.Name)
skaters$Team <- as.character(skaters$Team)
skaters$PP <- as.factor(as.integer(skaters$PP))
skaters$Line <- as.factor(as.integer(skaters$Line))

# Remove Rows where 'Proj.FP' = 0. The number of observations is now 30,471. -----------------------------------------
skaters <- subset(skaters, Proj.FP != 0)

# Sub-setting data sets by position to prepare for machine learning. -----------------------------------------
wings <- subset(skaters, Pos != "C" & Pos != "D")
centers <- subset(skaters, Pos != "D" & Pos != "W")
defensemen <- subset(skaters, Pos != "C" & Pos != "W")

# Save and export 'goalies' and 'skaters' data frames as excel file on desktop -----------
# This is the data frame we will test/train on.
install.packages('writexl')
library(writexl)
write_xlsx(goalies, 'C:\\Users\\jcole\\Documents\\DATA670\\machine_learning_players\\goalies_ml.xlsx')
write_xlsx(wings, 'C:\\Users\\jcole\\Documents\\DATA670\\machine_learning_players\\wings_ml.xlsx')
write_xlsx(centers, 'C:\\Users\\jcole\\Documents\\DATA670\\machine_learning_players\\centers_ml.xlsx')
write_xlsx(defensemen, 'C:\\Users\\jcole\\Documents\\DATA670\\machine_learning_players\\defensemen_ml.xlsx')

# Use 'dplyr' to stack 'goalies' and 'skaters' into 'players' data --------
library(dplyr)
players <- bind_rows(goalies, skaters)
apply(players, 2, function(players) sum(is.na(players)))
players$Line[is.na(players$Line)] <- 1
players$PP[is.na(players$PP)] <- 1
players[is.na(players)] <- 0

write.csv(players, 'C:\\Users\\jcole\\Documents\\DATA670\\machine_learning_players\\players.csv')

# Add columns to calculate mean absolute error -----------------------------------------
players$variance <- with(players, abs(Actual.FP - Proj.FP))

# Data Visualizations.
library(ggplot2)
library(lattice)
library(data.table)

boxplot(variance ~ Pos, data = players)

# Scatter Plot for Season average fantasy points scored versus actual points scored.
reg_g <- lm(S.FP ~ Actual.FP, data = goalies)
plot(goalies$S.FP, goalies$Actual.FP)
abline(reg_g, col = "blue")
ggplot(players, aes(x = S.FP, y = Actual.FP, group = Pos)) + geom_point(aes(color = Pos)) + scale_color_manual(values = c('Blue', 'Green', 'Red', 'Yellow')) + geom_smooth(method = "lm", se = FALSE) + scale_x_continuous(breaks = c(0, 10, 20, 30, 40, 50, 60)) + scale_y_continuous(breaks = c(0, 10, 20, 30, 40, 50, 60))

# Bar plot of top 5 Goalie scorers and top 10 Skater scorers.
goalies_top_5 <- data.frame(goalies$Player.Name, goalies$Pos, goalies$Actual.FP)
skaters_top_10 <- data.frame(skaters$Player.Name, skaters$Pos, skaters$Actual.FP)
goalies_top_5 <- aggregate(goalies.Actual.FP ~ goalies.Player.Name + goalies$Pos, data = goalies_top_5, FUN = sum)
skaters_top_10 <- aggregate(skaters.Actual.FP ~ skaters.Player.Name + skaters$Pos, data = skaters_top_10, FUN = sum)

goalies_top_5 <- goalies_top_5[order(goalies_top_5$goalies.Actual.FP, decreasing = TRUE)[1:5], ]
skaters_top_10 <- skaters_top_10[order(skaters_top_10$skaters.Actual.FP, decreasing = TRUE)[1:10], ]
goalies_top_5
goalies_top_5_bar <- c(963.2, 896.0, 892.0, 838.4, 824.8)
barplot(goalies_top_5_bar, main = "Top 5 Goalie FDP Scorers", xlab = "Player Name", ylab = "Total Fantasy Points Scored", names.arg = c("Juuse Saros", "Igor Shesterkin", "Jacob Markstrom", "Frederik Andersen", "Thatcher Demko"), col = "blue")
skaters_top_10
skaters_top_10_bar <- c(1246.7, 1099.0, 1078.0, 1060.0, 1059.7, 1042.4, 1032.0, 1006.3, 1002.2, 994.4)
barplot(skaters_top_10_bar, main = " Top 10 Skater FDP Scorers", xlab = "Player Name", ylab = "Total Fantasy Points Scored", 
        names.arg = c("A.Matthews (C)", "C.McDavid (C)", "K.Connor (W)", "A.Ovechkin (W)", "L.Draisaitl (C)", 
                      "K.Kaprizov (W)", "J.Gaudreau (W)", "J.Huberdeau (W)", "D.Pastrnak (W)", "M.Rantanen (W)"), col = "green", las = 2, cex.names = 0.6)

# Average Fantasy Points Scored per position.
players_mean <- aggregate(Actual.FP ~ Pos, data = players, FUN = mean)
players_mean
players_pie <- c(17.59, 9.01, 7.66, 8.79)
pie(players_pie, labels = c("G - 17.59", "C - 9.01", "D - 7.66", "W - 8.79"),main = "Average Points Scored by Position")
