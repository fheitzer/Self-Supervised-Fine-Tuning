#Bash script which downloads the datasets and puts them in the right directory structure

# Check if data is already downloaded

cd ..
cd ..
cd data

# Get Data and labels for every year
wget https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip
wget

# Unzip
unzip ISIC_2020_Training_JPEG.zip
unzip

# Move Everything into place
mv


# Remove old files
rm ISIC_2020_Training_JPEG.zip

exit
