//
//  StarRatingView.swift
//  EDEN
//
//  Created by Alana Kumar on 1/5/2026.
//

import SwiftUI

struct StarRatingView: View {
    @Binding var rating: Double
    let maxRating = 5

    var body: some View {
        HStack {
            ForEach(1...maxRating, id: \.self) { index in
                Image(systemName: starType(for: index))
                    .foregroundColor(.yellow)
                    .onTapGesture {
                        rating = Double(index)
                    }
                    .onLongPressGesture {
                        rating = Double(index) - 0.5
                    }
            }
        }
    }

    func starType(for index: Int) -> String {
        if rating >= Double(index) {
            return "star.fill"
        } else if rating >= Double(index) - 0.5 {
            return "star.leadinghalf.filled"
        } else {
            return "star"
        }
    }
}

