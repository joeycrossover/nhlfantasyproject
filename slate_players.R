# DATA 670 - Capstone, UMGC 2022
# Written by Joseph Coleman
# Last updated March 2, 2022

# The purpose of this code is to pre-process, clean, and join the given slate's player lists 
# from Daily Fantasy Nerd and FanDuel.
# This will allow us to include FanDuel's unique Player ID's that are needed for lineup importing.
# The trained ML models will optimize on the data frame built in this script.

# Directory Set-up --------------------------------------------------------
setwd("C:\\Users\\jcole\\Documents\\DATA670\\slate_player_lists")
dir()
library(dplyr)

# Loading Data Sets -------------------------------------------------------
dg <- read.csv(file="dailygoalies.csv", head=TRUE, sep=",", as.is=FALSE)
ds <- read.csv(file="dailyskaters.csv", head=TRUE, sep=",", as.is=FALSE)
fd <- read.csv(file="fdplayers.csv", head=TRUE, sep=",", as.is=FALSE)

# 'dg' data set pre-processing and clean-up -------------------------------
apply(dg, 2, function(dg) sum(is.na(dg)))
dg$Likes[is.na(dg$Likes)] <- 0
dg$Likes <- as.integer(dg$Likes)
dg$Inj <- as.character(dg$Inj)
dg$Inj[is.na(dg$Inj)] <- ""
dg$Vegas.Odds.Win <- gsub("%", "", as.character(dg$Vegas.Odds.Win))
dg$Vegas.Odds.Win <- as.numeric(dg$Vegas.Odds.Win)
dg$Salary <- as.numeric(dg$Salary)
dg$Opp <- gsub("@","", as.character(dg$Opp))
dg$Player.Name <- as.character(dg$Player.Name)
dg$Team <- as.character(dg$Team)
dg$Player.Name <- gsub("Calvin Petersen", "Cal Petersen", as.character(dg$Player.Name))
dg$Player.Name <- gsub("Nicolas Daws", "Nico Daws", as.character(dg$Player.Name))

# 'ds' data set pre-processing and clean-up --------------------------------
apply(ds, 2, function(ds) sum(is.na(ds)))
ds$Likes[is.na(ds$Likes)] <- 0
ds$Likes <- as.integer(ds$Likes)
ds$Inj <- as.character(ds$Inj)
ds$Inj[is.na(ds$Inj)] <- ""
ds <- na.omit(ds)
ds$Rest <- gsub("n/a", "", as.character(ds$Rest))
ds$Rest <- as.integer(as.character(ds$Rest))
ds$Salary <- as.numeric(ds$Salary)
ds$Opp <- gsub("@","", as.character(ds$Opp))
ds$Player.Name <- as.character(ds$Player.Name)
ds$Team <- as.character(ds$Team)
ds$PP <- as.factor(as.integer(ds$PP))
ds$Line <- as.factor(as.integer(ds$Line))

# Remove rows where 'Proj.FP' = 0 -----------------------------------------
dg <- subset(dg, Proj.FP != 0)
ds <- subset(ds, Proj.FP != 0)

# Sub-setting data sets by position to prepare for machine learning. -----------------------------------------
dw <- subset(ds, Pos != "C" & Pos != "D")
dc <- subset(ds, Pos != "D" & Pos != "W")
dd <- subset(ds, Pos != "C" & Pos != "W")

# Replacing Names per position so that FD list matches with DFN. --------------------------
dc$Player.Name <- gsub("Aleksander Barkov Jr.", "Aleksander Barkov", as.character(dc$Player.Name))
dc$Player.Name <- gsub("Theodor Blueger", "Teddy Blueger", as.character(dc$Player.Name))
dc$Player.Name <- gsub("Alexey Lipanov", "Alexei Lipanov", as.character(dc$Player.Name))
dc$Player.Name <- gsub("Artem Anisimov", "Artyom A Anisimov", as.character(dc$Player.Name))
dc$Player.Name <- gsub("Mikey Eyssimont", "Michael Eyssimont", as.character(dc$Player.Name))
dc$Player.Name <- gsub("Jean-Christophe Beaudin", "J.C. Beaudin", as.character(dc$Player.Name))
dc$Player.Name <- gsub("Alexander Wennberg", "Alex Wennberg", as.character(dc$Player.Name))

dd$Player.Name <- gsub("Zachary Werenski", "Zach Werenski", as.character(dd$Player.Name))
dd$Player.Name <- gsub("Mathew Dumba", "Matt Dumba", as.character(dd$Player.Name))
dd$Player.Name <- gsub("Joshua Brown", "Josh Brown", as.character(dd$Player.Name))
dd$Player.Name <- gsub("Alex Petrovic", "Alexander Petrovic", as.character(dd$Player.Name))
dd$Player.Name <- gsub("Mitchell Vande Sompel", "Mitch Vande Sompel", as.character(dd$Player.Name))
dd$Player.Name <- gsub("Will Reilly", "William Reilly", as.character(dd$Player.Name))
dd$Player.Name <- gsub("Nathan Clurman", "Nate Clurman", as.character(dd$Player.Name))
dd$Player.Name <- gsub("Maxwell Gildon", "Max Gildon", as.character(dd$Player.Name))
dd$Player.Name <- gsub("William Borgen", "Will Borgen", as.character(dd$Player.Name))
dd$Player.Name <- gsub("Artyom Zub", "Artem Zub", as.character(dd$Player.Name))

dw$Player.Name <- gsub("Matthew Boldy", "Matt Boldy", as.character(dw$Player.Name))
dw$Player.Name <- gsub("Egor Sharangovich", "Yegor Sharangovich", as.character(dw$Player.Name))
dw$Player.Name <- gsub("Gerald Mayhew", "Gerry Mayhew", as.character(dw$Player.Name))
dw$Player.Name <- gsub("Joe Gambardella", "Joseph Gambardella", as.character(dw$Player.Name))
dw$Player.Name <- gsub("Frederik Gauthier", "Freddy Gauthier", as.character(dw$Player.Name))
dw$Player.Name <- gsub("Nicholas Caamano", "Nick Caamano", as.character(dw$Player.Name))
dw$Player.Name <- gsub("Zach Jordan", "Zac Jordan", as.character(dw$Player.Name))
dw$Player.Name <- gsub("Grigory Denisenko", "Grigori Denisenko", as.character(dw$Player.Name))
dw$Player.Name <- gsub("Alexander Nylander", "Alex Nylander", as.character(dw$Player.Name))
dw$Player.Name <- gsub("Egor Korshkov", "Yegor Korshkov", as.character(dw$Player.Name))
dw$Player.Name <- gsub("Cameron Morrison", "Cam Morrison", as.character(dw$Player.Name))
dw$Player.Name <- gsub("Nicholas Ritchie", "Nick Ritchie", as.character(dw$Player.Name))
dw$Player.Name <- gsub("Tim Stutzle", "Tim Stuetzle", as.character(dw$Player.Name))
dw$Player.Name <- gsub("Vincent Hinostroza", "Vinnie Hinostroza", as.character(dw$Player.Name))

# Clean FanDuel Player List and join player IDs on each data set for each position. --------------------------------
fd$Nickname <- as.character(fd$Nickname)
fd$Id <- as.character(fd$Id)

gfd <- subset(fd, (Nickname %in% dg$Player.Name))
gfd <- gfd[, c("Id", "Nickname")]
gfd = merge(dg, gfd, by.x = "Player.Name",
           by.y = "Nickname")
gfd <- gfd %>% select(Id, everything())

wfd <- subset(fd, (Nickname %in% dw$Player.Name))
wfd <- wfd[, c("Id", "Nickname")]
wfd = merge(dw, wfd, by.x = "Player.Name",
            by.y = "Nickname")
wfd <- wfd %>% select(Id, everything())

cfd <- subset(fd, (Nickname %in% dc$Player.Name))
cfd <- subset(cfd, Position != "D")
cfd <- cfd[, c("Id", "Nickname")]
cfd = merge(dc, cfd, by.x = "Player.Name",
            by.y = "Nickname")
cfd <- cfd %>% select(Id, everything())

dfd <- subset(fd, (Nickname %in% dd$Player.Name))
dfd <- dfd[, c("Id", "Nickname")]
dfd = merge(dd, dfd, by.x = "Player.Name",
            by.y = "Nickname")
dfd <- dfd %>% select(Id, everything())

# Save and export joined data set as xlsx file onto desktop --------------
install.packages('writexl')
library(writexl)
write_xlsx(gfd, 'C:\\Users\\jcole\\Documents\\DATA670\\slate_player_lists\\gfd.xlsx')
write_xlsx(wfd, 'C:\\Users\\jcole\\Documents\\DATA670\\slate_player_lists\\wfd.xlsx')
write_xlsx(cfd, 'C:\\Users\\jcole\\Documents\\DATA670\\slate_player_lists\\cfd.xlsx')
write_xlsx(dfd, 'C:\\Users\\jcole\\Documents\\DATA670\\slate_player_lists\\dfd.xlsx')

slateplayers <- bind_rows(cfd, wfd, dfd, gfd)
apply(slateplayers, 2, function(slateplayers) sum(is.na(slateplayers)))
slateplayers$Line[is.na(slateplayers$Line)] <- 1
slateplayers$PP[is.na(slateplayers$PP)] <- 1
slateplayers[is.na(slateplayers)] <- 0

write.csv(slateplayers, 'C:\\Users\\jcole\\Documents\\DATA670\\slate_player_lists\\slateplayers.csv')
