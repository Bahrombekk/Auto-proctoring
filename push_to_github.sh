#!/bin/bash
git fetch --all
cd ~/Desktop/Auto-proctoring || {
    echo "Papkaga o'tib bo'lmadi. Yo'lni tekshiring."
    exit 1
}

# Branchlar ro'yxatini olish
branches=($(git branch -r | grep -v '\->' | sed 's/origin\///'))

echo "=== GitHub branchlar ro'yxati ==="
for i in "${!branches[@]}"; do
    echo "$i) ${branches[$i]}"
done

# Foydalanuvchidan tanlov so'rash
read -p "Branch raqamini tanlang (masalan, 0): " choice

# Raqam tekshiruvi
if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -ge "${#branches[@]}" ]; then
    echo "Noto'g'ri tanlov. Dastur yakunlandi."
    exit 1
fi

branch=${branches[$choice]}
echo "Tanlangan branch: $branch"

# Git add va commit
git add .

git commit -m "Avtomatik push: $(date)" || {
    echo "Hech qanday o'zgarish yo'q yoki commitda xato yuz berdi."
    exit 1
}

# Push
git push origin "$branch" || {
    echo "Push qilishda xato yuz berdi. Autentifikatsiyani tekshiring yoki git pull qiling."
    exit 1
}

echo "✅ Barcha o'zgarishlar GitHub branch '$branch' ga muvaffaqiyatli yuklandi!"
