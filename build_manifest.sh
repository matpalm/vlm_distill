find PetImages/Cat/ -type f | sort | head -n100 | perl -ne'chomp;print "$_\tcat\ttrain\n"' > manifest.tsv
find PetImages/Cat/ -type f | sort | head -n200 | tail -n100 | perl -ne'chomp;print "$_\tcat\tvalidate\n"' >> manifest.tsv
find PetImages/Dog/ -type f | sort | head -n100 | perl -ne'chomp;print "$_\tdog\ttrain\n"' >> manifest.tsv
find PetImages/Dog/ -type f | sort | head -n200 | tail -n100 | perl -ne'chomp;print "$_\tdog\tvalidate\n"' >> manifest.tsv