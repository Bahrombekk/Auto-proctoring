#!/bin/bash

# Git remote holatini yangilash
git fetch --all

# Loyihaning joylashgan katalogiga o‘tish
cd ~/Desktop/Auto-proctoring || {
    echo "❌ Papkaga o'tib bo'lmadi. Yo'lni tekshiring."
    exit 1
}

# Remote branchlar ro'yxatini olish
branches=($(git branch -r | grep -v '\->' | sed 's/origin\///'))

echo "=== GitHub branchlar ro'yxati ==="
for i in "${!branches[@]}"; do
    echo "$i) ${branches[$i]}"
done
echo "${#branches[@]}) ➕ Yangi branch yaratish"

# Foydalanuvchidan tanlov
read -p "Branch raqamini tanlang yoki yangi branch yaratish uchun raqam kiriting (${#branches[@]}): " choice

# Yangi branch yaratish
if [ "$choice" -eq "${#branches[@]}" ]; then
    read -p "Yangi branch nomini kiriting: " new_branch

    # Branch nomi bo'sh emasligini tekshirish
    if [[ -z "$new_branch" ]]; then
        echo "❌ Branch nomi bo'sh bo'lishi mumkin emas."
        exit 1
    fi

    git checkout -b "$new_branch"
    git push -u origin "$new_branch"

    branch="$new_branch"
else
    # Raqamni tekshirish
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -ge "${#branches[@]}" ]; then
        echo "❌ Noto'g'ri tanlov. Dastur yakunlandi."
        exit 1
    fi

    branch=${branches[$choice]}
    git checkout "$branch"
    git pull origin "$branch"
fi

# Git add va commit
git add .

git commit -m "Avtomatik push: $(date)" || {
    echo "⚠️ Hech qanday o'zgarish yo'q yoki commitda xato yuz berdi."
    exit 1
}

# Push
git push origin "$branch" || {
    echo "❌ Push qilishda xato yuz berdi. Autentifikatsiyani yoki git pull holatini tekshiring."
    exit 1
}

echo "✅ Barcha o'zgarishlar GitHub branch '$branch' ga muvaffaqiyatli yuklandi!"
